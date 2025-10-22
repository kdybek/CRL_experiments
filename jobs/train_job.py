from losses import contrastive_loss, contrastive_loss_same_trajectories
import sklearn.manifold
import gin
import matplotlib.pyplot as plt
import time
import numpy as np

import matplotlib.pyplot as plt

from datasets.utils import tokenize_pair
from datasets.utils import DataLoader
from datasets.contrastive_diff_len import DatasetCRTR, DatasetSameTrajGeom, DatasetSameTrajUnif, DatasetOBBT
from search.value_function import ValueEstimator
from search.solve_job import SolveJob

import torch
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import spearmanr


from pathlib import Path


@gin.configurable
class TrainJob():
    def __init__(
        self,
        loggers,
        train_steps,
        batch_size,
        lr,
        model_type,
        metric,
        search_shuffles,
        output_dir,
        test_interval,
        metric_log_interval=100,
        do_eval=True,
        tokenizer=tokenize_pair,
        eval_job_class=None,
        checkpoint_path=None,
    ):
        self.loggers = loggers
        self.train_steps = train_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self.model = model_type().to(self.device)
        self.metric_log_interval = metric_log_interval
        self.test_interval = test_interval

        self.batch_size = batch_size
        self.lr = lr
        self.do_eval = do_eval
        self.eval_job_class = eval_job_class
        self.metric = metric
        self.output_dir = output_dir
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.checkpoint_path is not None:
            self.read_checkpoint(self.checkpoint_path)

        self.search_shuffles = search_shuffles
        self.model.to(self.device)

    def save_checkpoint(self, step):
        model_checkpoint_path = f"{self.output_dir}/{step}/model.pt"
        optimizer_checkpoint_path = f"{self.output_dir}/{step}/optimizer"
        path = Path(model_checkpoint_path)
        path_opt = Path(optimizer_checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        torch.save(self.optimizer.state_dict(), path_opt)

    def read_checkpoint(self, path):
        model_checkpoint_path = f"{path}/model.pt"
        model_checkpoint = torch.load(
            model_checkpoint_path, weights_only=True, map_location=torch.device(self.device))
        self.model.load_state_dict(model_checkpoint)

        optimizer_checkpoint_path = f"{path}/optimizer"
        optimizer_checkpoint = torch.load(
            optimizer_checkpoint_path, weights_only=True, map_location=torch.device(self.device))
        self.optimizer.load_state_dict(optimizer_checkpoint)

    def gen_plot_distances(self, step):
        value_estimator = ValueEstimator(self.model, self.metric)
        all_distances = []
        for i, s in enumerate(self.test_trajectories):
            distances = value_estimator.get_solved_distance_batch(s, s[-1])
            all_distances.append(distances.cpu().numpy())

        all_distances = np.array(all_distances).mean(axis=0)
        plt.plot(np.arange(len(all_distances)), all_distances)
        self.loggers.log_figure(f'avg distances solved', step, plt.gcf())
        plt.clf()

    def gen_plot_TSNE(self, test_trajectories, step):
        MAX_TRAJ_TO_ANALYSE = 20
        TRAJ_LEN = 10

        filtered_traj = [traj for traj in test_trajectories if len(traj) >= TRAJ_LEN]
        selected_traj = filtered_traj[:MAX_TRAJ_TO_ANALYSE]
        num_selected = len(selected_traj)
        all_embeddings = []

        for i in range(num_selected):
            trajectory = test_trajectories[i]
            trajectory = trajectory.reshape(trajectory.shape[0], -1)
            trajectory = trajectory[:TRAJ_LEN]
            embeddings_double = self.model(trajectory).detach().cpu().numpy()
            all_embeddings.append(embeddings_double)

        all_embeddings = np.concatenate(all_embeddings)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)

        embeddings_2d = tsne.fit_transform(all_embeddings)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        distinct_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                           '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
                           '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f',
                           '#e5c494', '#b3b3b3', '#8dd3c7', '#bebada', '#fb8072'] * 2

        for i in range(num_selected):
            ax.scatter(embeddings_2d[i*TRAJ_LEN:(i+1)*TRAJ_LEN, 0], embeddings_2d[i*TRAJ_LEN:(i+1)*TRAJ_LEN, 1],
                       alpha=0.6, s=20, color=distinct_colors[i])

        plt.tight_layout()
        self.loggers.log_figure("t-sne reps", step, plt.gcf())
        plt.clf()

    def gen_plot_1(self, test_trajectories, step):
        for traj in test_trajectories:
            with torch.no_grad():
                traj = traj.to(self.device)
                psi = self.model(traj)
                if self.metric == 'mrn':
                    psi = psi[..., psi.shape[-1] // 2:]
                psi = psi.cpu()
                traj = traj.to('cpu')
                del traj

            tsne = sklearn.manifold.TSNE(n_components=2, perplexity=5)
            psi = tsne.fit_transform(psi)

            plt.scatter(psi[:, 0], psi[:, 1], marker='.',
                        c=np.arange(len(psi)), cmap='Reds')

        plt.gca().set_aspect('equal')
        self.loggers.log_figure("All reps", step, plt.gcf())
        plt.clf()

    def gen_plot_2(self, test_trajectories, step):
        for i, s in enumerate(test_trajectories):
            if i == 4:
                break

            with torch.no_grad():
                s = s.to(self.device)
                psi = self.model(s)

                if self.metric == 'mrn':
                    psi = psi[..., psi.shape[-1] // 2:]

                psi = psi.cpu()
                s = s.to('cpu')
                del s

            tsne = sklearn.manifold.TSNE(n_components=2, perplexity=5)
            psi = tsne.fit_transform(psi)
            beginning = psi[0]
            end = psi[-1]

            c_vec = plt.rcParams['axes.prop_cycle'].by_key()['color']
            plt.text(psi[0, 0], psi[0, 1], '$x_0$',
                     ha='center', va='bottom', fontsize=16)
            plt.text(psi[-1, 0], psi[-1, 1], '$x_T$',
                     ha='center', va='bottom', fontsize=16)

            plt.plot(psi[:, 0], psi[:, 1], '-', c=c_vec[0], linewidth=1, alpha=0.1)
            plt.scatter(psi[:, 0], psi[:, 1], c=np.arange(len(psi)), cmap='plasma')

            n_wypt = 5

            vec = np.linspace(beginning, end, n_wypt)
            plt.scatter(vec[:, 0], vec[:, 1], c=np.arange(len(vec)), cmap='Greys')

            plt.gca().set_aspect('equal')
            self.loggers.log_figure(f'plot {i}', step, plt.gcf())
            plt.clf()

    def gen_plot_PCA(self, test_trajectories, step):
        MAX_TRAJ_TO_ANALYSE = 20
        TRAJ_LEN = 10

        filtered_traj = [traj for traj in test_trajectories if len(traj) >= TRAJ_LEN]
        selected_traj = filtered_traj[:MAX_TRAJ_TO_ANALYSE]
        num_selected = len(selected_traj)
        all_embeddings = []

        for i in range(num_selected):
            trajectory = test_trajectories[i]
            trajectory = trajectory.reshape(trajectory.shape[0], -1)
            trajectory = trajectory[:TRAJ_LEN]
            embeddings_double = self.model(trajectory).detach().cpu().numpy()
            all_embeddings.append(embeddings_double)

        all_embeddings = np.concatenate(all_embeddings)

        pca = PCA(n_components=2)

        embeddings_2d = pca.fit_transform(all_embeddings)

        exp_var = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
        self.loggers.log_scalar("PCA explained variance", step, exp_var)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        distinct_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                           '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
                           '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f',
                           '#e5c494', '#b3b3b3', '#8dd3c7', '#bebada', '#fb8072'] * 2

        for i in range(num_selected):
            ax.scatter(embeddings_2d[i*TRAJ_LEN:(i+1)*TRAJ_LEN, 0], embeddings_2d[i*TRAJ_LEN:(i+1)*TRAJ_LEN, 1],
                       alpha=0.6, s=20, color=distinct_colors[i])

        plt.tight_layout()
        self.loggers.log_figure("PCA reps", step, plt.gcf())
        plt.clf()

    def gen_plot_monotonicity(self, test_trajectories, step):
        value_estimator = ValueEstimator(self.model, self.metric)
        correlations = []
        for i, s in enumerate(test_trajectories):
            s = s.to(self.device)
            distances = value_estimator.get_solved_distance_batch(s, s[-1]).to('cpu')
            s = s.to('cpu')
            del s
            correlation = spearmanr(distances.cpu(), np.arange(
                len(distances.cpu()), 0, -1)).statistic
            correlations.append(correlation)
            if i < 4:

                self.loggers.log_scalar(f'correlation {i}', step, correlation)

                plt.plot(np.arange(distances.cpu().shape[-1]), distances.cpu())

                self.loggers.log_figure(f'monotonicity {i}', step, plt.gcf())
                plt.clf()

        self.loggers.log_scalar('correlation', step, sum(
            correlations)/len(correlations))

    def log_metrics(self, step):
        for name, value in self.metrics.items():
            self.loggers.log_scalar(name, step, value)

    def test_and_log(self, step):
        with torch.no_grad():
            self.gen_plot_monotonicity(
                test_trajectories=self.test_trajectories, step=step)
            self.gen_plot_TSNE(test_trajectories=self.test_trajectories, step=step)
            self.gen_plot_PCA(test_trajectories=self.test_trajectories, step=step)
            self.gen_plot_1(test_trajectories=self.test_trajectories, step=step)
            self.gen_plot_2(test_trajectories=self.test_trajectories, step=step)

            for shuffles in self.search_shuffles:
                eval_job = SolveJob(
                    loggers=self.loggers, network=self.model, step=step, metric=self.metric, shuffles=shuffles)
                eval_job.execute()
                break


@gin.configurable
class TrainJobCRTR(TrainJob):
    def __init__(self, train_path, test_path, n_test_traj, gamma, repetition_rate, **kwargs):
        super().__init__(**kwargs)
        self.dataset = DatasetCRTR(
            path=train_path, gamma=gamma, repetition_rate=repetition_rate, device=self.device)

        self.train_dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size)

        self.test_dataset = DatasetCRTR(
            path=test_path, gamma=0.9, repetition_rate=1, device=self.device)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size)

        self.test_trajectories = [self.dataset._get_trajectory()
                                  for _ in range(n_test_traj)]

    def execute(self):
        step = 1
        while step <= self.train_steps:
            for data in self.train_dataloader:
                self.model.train()

                self.optimizer.zero_grad()
                x0, xT = data
                psi_0 = self.model(x0)
                psi_T = self.model(xT)
                loss, self.metrics = contrastive_loss(
                    psi_0, psi_T, distance_fun=self.metric)
                loss.backward()

                self.optimizer.step()

                if step % self.metric_log_interval == 0:
                    self.log_metrics(step)
                    self.loggers.log_scalar('step', step, step)

                if step % self.test_interval == 0:
                    self.test_and_log(step)
                    self.save_checkpoint(step)

                del data

                step += 1
                if step > self.train_steps:
                    break

        self.save_checkpoint('final')


@gin.configurable
class TrainJobSameTraj(TrainJob):
    def __init__(self, train_path, dist, test_path, n_test_traj, gamma, n_negatives, gamma_negative=None, **kwargs):
        assert dist in ['geom', 'unif']
        assert (dist == 'geom' and gamma_negative is not None) or (
            dist == 'unif' and gamma_negative is None)

        super().__init__(**kwargs)
        if dist == 'geom':
            self.dataset = DatasetSameTrajGeom(
                path=train_path, gamma=gamma, gamma_negative=gamma_negative, n_negatives=n_negatives, device=self.device)
        else:
            self.dataset = DatasetSameTrajUnif(
                path=train_path, gamma=gamma, n_negatives=n_negatives, device=self.device)

        self.train_dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size)

        self.test_dataset = DatasetCRTR(
            path=test_path, gamma=0.9, repetition_rate=1, device=self.device)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size)

        self.test_trajectories = [self.dataset._get_trajectory()
                                  for _ in range(n_test_traj)]

    def execute(self):
        step = 1
        while step <= self.train_steps:
            for data in self.train_dataloader:
                self.model.train()

                self.optimizer.zero_grad()
                x0, xT, xR = data
                psi_0 = self.model(x0)
                psi_T = self.model(xT)
                psi_R = self.model(xR)
                loss, self.metrics = contrastive_loss_same_trajectories(
                    psi_0, psi_T, psi_R)
                loss.backward()

                self.optimizer.step()

                if step % self.metric_log_interval == 0:
                    self.log_metrics(step)
                    self.loggers.log_scalar('step', step, step)

                if step % self.test_interval == 0:
                    self.test_and_log(step)
                    self.save_checkpoint(step)

                del data

                step += 1
                if step > self.train_steps:
                    break

        self.save_checkpoint('final')


@gin.configurable
class TrainJobOBBT(TrainJob):
    def __init__(self, train_path, test_path, n_test_traj, mbbt=False, **kwargs):
        super().__init__(**kwargs)
        self.dataset = DatasetOBBT(path=train_path, device=self.device)
        self.mbbt = mbbt

        self.train_dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size)

        self.test_dataset = DatasetCRTR(
            path=test_path, gamma=0.9, repetition_rate=1, device=self.device)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size)

        self.test_trajectories = [self.dataset._get_trajectory()
                                  for _ in range(n_test_traj)]

    def obbt_loss(self, psi_0_concat, psi_T_concat, lens):
        psi_0s = torch.split(psi_0_concat, lens, dim=0)
        psi_Ts = torch.split(psi_T_concat, lens, dim=0)

        loss = 0
        for psi_0, psi_T in zip(psi_0s, psi_Ts, strict=True):
            small_loss, _ = contrastive_loss(
                psi_0, psi_T, distance_fun=self.metric, weight_matrix=None)

            small_loss = small_loss / psi_0.shape[0]
            loss = loss + small_loss

        self.metrics = {'loss': loss.item()}

        return loss

    def _create_weight_matrix(self, traj_len, gamma):
        idx = torch.arange(traj_len - 1)
        diff = idx.unsqueeze(0) - idx.unsqueeze(1) + 1
        abs_diff = torch.abs(diff)
        W = (1 / gamma) ** abs_diff

        return W

    def execute(self):
        step = 1
        while step <= self.train_steps:
            for data in self.train_dataloader:
                self.model.train()

                states, goals, lens = data
                psi_0_concat = self.model(states)
                psi_T_concat = self.model(goals)
                # traj_len = states.shape[0] + 1
                # W = self._create_weight_matrix(traj_len, 0.99).to(self.device)

                if self.mbbt:
                    loss, self.metrics = contrastive_loss(
                        psi_0_concat, psi_T_concat, distance_fun=self.metric)

                else:
                    loss = self.obbt_loss(psi_0_concat, psi_T_concat, lens)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % self.metric_log_interval == 0:
                    self.log_metrics(step)
                    self.loggers.log_scalar('step', step, step)

                if step % self.test_interval == 0:
                    self.test_and_log(step)
                    self.save_checkpoint(step)

                del data

                step += 1
                if step > self.train_steps:
                    break
