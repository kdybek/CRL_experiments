import re
import gin
import joblib
import os
import numpy as np
import torch


def get_dataset_stats(directory):
    num_traj = 0
    file_paths = []

    for filename in os.listdir(directory):
        match = re.match(r"^(.*)_trajectories\.pkl$", filename)
        if match:
            prefix = match.group(1)
            len_filename = f"{prefix}_lens.pkl"
            num_traj += 1
            file_paths.append((directory, filename, len_filename))

    file_paths = sorted(file_paths)
    return num_traj, file_paths


@gin.configurable
class ContrastiveDatasetDiffLen():
    def __init__(self, path, device='cpu'):
        self.num_traj, self.file_paths = get_dataset_stats(path)
        self.path = path

        self.buffer = None
        self.buffer_ind = 0
        self.file_ind = 0

        self.device = device

        assert device in ['cpu', 'cuda']

    def _read_next_file(self):
        self.file_ind = np.random.randint(len(self.file_paths))

        current_dir, current_file, curren_len_file = self.file_paths[self.file_ind]
        current_path = current_dir + '/' + current_file
        current_len_path = current_dir + '/' + curren_len_file

        with open(current_path, 'rb') as f:
            self.buffer = joblib.load(f)

        with open(current_len_path, 'rb') as f:
            self.buffer_lens = joblib.load(f)

        self.buffer = self.buffer[self.buffer_lens != 0]
        self.buffer_lens = self.buffer_lens[self.buffer_lens != 0]

        permutation = np.random.permutation(len(self.buffer))
        self.buffer = self.buffer[permutation]
        self.buffer_lens = self.buffer_lens[permutation]

        self.buffer_ind = 0

    def _get_trajs(self, n_traj):
        if self.buffer is None or self.buffer_ind + n_traj > len(self.buffer):
            self._read_next_file()

        self.buffer_ind += n_traj
        current_buffer, current_lens = self.buffer[self.buffer_ind - n_traj:self.buffer_ind].to(
            torch.float32), self.buffer_lens[self.buffer_ind - n_traj:self.buffer_ind]

        return current_buffer, current_lens

    def _get_trajectory(self):
        traj, len = self._get_trajs(1)
        traj = traj.squeeze()
        traj = traj.flatten(1)
        len = len.item()

        traj = traj[:len]
        return traj.to(self.device)


@gin.configurable
class DatasetCRTR(ContrastiveDatasetDiffLen):
    def __init__(self, path, gamma=0.9, max_horizon=200, double_batch=1, device='cpu'):
        super().__init__(path, device)
        self.gamma = gamma
        self.max_horizon = max_horizon
        self.double_batch = double_batch

    def _get_batch(self, batch_size):
        with torch.no_grad():
            if self.double_batch:
                trajs, lens = self._get_trajs(int(batch_size / self.double_batch))

                if len(trajs.shape) == 3:
                    trajs = trajs.repeat(int(self.double_batch) + 1, 1, 1)[:batch_size]
                elif len(trajs.shape) == 4:
                    trajs = trajs.repeat(int(self.double_batch) +
                                         1, 1, 1, 1)[:batch_size]

                lens = lens.repeat(int(self.double_batch) + 1)[:batch_size]
            else:
                trajs, lens = self._get_trajs(batch_size)

            assert len(trajs.shape) in [3, 4]
            assert len(trajs) > 0
            weights = torch.zeros((trajs.shape[0], trajs.shape[1]))
            mask = torch.arange(len(trajs[0])).unsqueeze(
                0).repeat(len(trajs), 1) < lens.unsqueeze(1).cpu()

            weights[mask] = 1

            i = torch.multinomial(weights.float(), num_samples=1).squeeze()

            horizon = lens - i - 1

            probs = self.gamma ** torch.arange(self.max_horizon).unsqueeze(
                0).repeat(len(trajs), 1).float()

            mask = torch.arange(self.max_horizon).repeat(
                len(trajs), 1) <= horizon.unsqueeze(1)
            probs *= mask.float()

            probs /= probs.sum(dim=1, keepdim=True)

            delta = torch.multinomial(probs, num_samples=1).squeeze()

            states = trajs[torch.arange(len(trajs)), i]
            goals = trajs[torch.arange(len(trajs)), i+delta]

            if len(trajs.shape) == 4:
                goals = goals.flatten(1)
                states = states.flatten(1)

            goals = goals.to(torch.float32).to(self.device)
            states = states.to(torch.float32).to(self.device)

        return states, goals


@gin.configurable
class DatasetSameTrajGeom(ContrastiveDatasetDiffLen):
    def __init__(self, path, gamma=0.9, gamma_negative=0.9, n_negatives=4, device='cpu'):
        super().__init__(path, device)
        self.gamma = gamma
        self.gamma_negative = gamma_negative
        self.n_negatives = n_negatives

    def _get_batch(self, batch_size):
        with torch.no_grad():
            trajs, lens = self._get_trajs(batch_size)

            assert len(trajs.shape) in [3, 4]
            assert len(trajs) > 0

            trajs = trajs.to(self.device)
            lens = lens.to(self.device)

            # ----- Sample state1 -----
            rand_vals1 = torch.rand(batch_size, device=self.device)
            id1 = (rand_vals1 * lens).floor().to(torch.int32)

            # ----- Sample state2 (gamma-discounted offset from state1) -----
            maximal_offsets = lens - id1 - 1  # shape (B,)
            maximal_len = int(max(maximal_offsets).item())
            powers = torch.arange(maximal_len + 1, device=self.device)
            gamma_powers = self.gamma ** powers

            mask = powers.unsqueeze(0) <= maximal_offsets.unsqueeze(1)
            probs = gamma_powers.unsqueeze(0).expand(batch_size, -1) * mask
            probs = probs / probs.sum(dim=1, keepdim=True)

            offset = torch.multinomial(probs, 1).squeeze(1)
            id2 = id1 + offset

            batch_indices = torch.arange(batch_size, device=self.device)
            states = trajs[batch_indices, id1]  # (B, D)
            goals = trajs[batch_indices, id2]  # (B, D)

            # ----- Decide how many to sample from start vs. end -----
            before_id1 = id1 + 1             # states 0 ... id1[i]
            after_id1 = lens - id1  # states id1[i] ... end

            p_start = before_id1 / (before_id1 + after_id1)

            how_many_biased_toward_start = torch.distributions.Binomial(
                total_count=self.n_negatives, probs=p_start
            ).sample().to(torch.long).to(self.device)  # use long for indexing

            offset_from_start = torch.multinomial(
                probs, self.n_negatives, replacement=True)  # (B, n_negatives)
            offset_from_end = torch.multinomial(
                probs, self.n_negatives, replacement=True)  # (B, n_negatives)

            id_start = offset_from_start
            id_end = lens.unsqueeze(1) - 1 - offset_from_end

            mask = torch.arange(self.n_negatives, device=self.device).unsqueeze(0)
            mask = mask < how_many_biased_toward_start.unsqueeze(1)

            rand_inds = torch.where(mask, id_start, id_end)

            random_states = trajs[batch_indices.unsqueeze(1), rand_inds]  # (B, n_negatives, D)

            if len(trajs.shape) == 4:
                states = states.flatten(1)
                goals = goals.flatten(1)
                random_states = random_states.flatten(2)

            states = states.to(torch.float32).to(self.device)
            goals = goals.to(torch.float32).to(self.device)
            random_states = random_states.to(torch.float32).to(self.device)

        return states, goals, random_states
