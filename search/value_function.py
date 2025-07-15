import os
import torch
import gin

@gin.configurable
class ValueEstimator:
    def __init__(self, model, metric, include_actions=False, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.distance = (lambda x, y: (((x-y)**2).sum(dim=1))**(0.5)) if metric == 'l22' else (lambda x, y: ((x-y)**2).sum(dim=1)) if metric == 'l2' else (lambda x, y: (torch.abs((x-y))).sum(dim=1)) if metric == 'l1' else (lambda x, y: -(torch.matmul(x, y.T))) if metric == 'dot' else None
        self.include_actions = include_actions

        if self.distance is None:
            raise ValueError()
        
    def construct_networks(self):
        if self.checkpoint_path is None:
            return

        if os.path.isfile(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        else:
            subdirs = [d for d in os.listdir(self.checkpoint_path) if os.path.isdir(os.path.join(self.checkpoint_path, d))]
            
            numeric_subdirs = sorted(
                [d for d in subdirs if d.isdigit()],
                key=lambda x: int(x),
                reverse=True
            )
            
            if not numeric_subdirs:
                raise ValueError("No valid checkpoint folders found in the directory.")
            
            latest_dir = os.path.join(self.checkpoint_path, numeric_subdirs[0])
            
            checkpoint_file = os.path.join(latest_dir, 'model.pt')  # assuming filename
            if not os.path.isfile(checkpoint_file):
                raise FileNotFoundError(f"No model.pt found in {latest_dir}")
            
            self.model.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
    
    def get_solved_distance(self, state_str, goal, action_in=None):
        num_state = torch.tensor(state_str).to(self.device)
        num_goal = torch.tensor(goal).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            goal_repr = self.model(num_goal.unsqueeze(0))
            state_repr = self.model(num_state.unsqueeze(0))

        distance = self.distance(state_repr, goal_repr)
        
        return distance.to('cpu')
    
    def get_solved_distance_batch(self, states, goal):
        num_goal = torch.tensor(goal)
        self.model.eval()
        with torch.no_grad():
            goal_repr = self.model(num_goal.unsqueeze(0).to(states.device))
            state_repr = self.model(states)

        distance = self.distance(state_repr, goal_repr)

        return distance.squeeze()





