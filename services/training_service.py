import docker
import os
from pathlib import Path
import csv

class TrainingService:
    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Error initializing Docker client: {e}")
            self.client = None
            
        self.base_dir = Path(os.getcwd()).absolute()
        self.dataset_root_search = self.base_dir / "dataset"
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)

    def _ensure_docker_client(self):
        """Try to initialize docker client if it doesn't exist"""
        if self.client is None:
            try:
                self.client = docker.from_env()
            except Exception as e:
                print(f"Error initializing Docker client: {e}")
                self.client = None


    def _find_dataset_path(self):
        """
        Recursively find the directory containing 'data.yaml'.
        Returns (path_to_dataset_folder, path_to_yaml_file)
        """
        for root, dirs, files in os.walk(self.dataset_root_search):
            if "data.yaml" in files:
                return Path(root), Path(root) / "data.yaml"
        return None, None

    def _create_docker_config(self, original_yaml_path: Path, dataset_folder: Path):
        """
        Reads the original data.yaml and creates a new data_docker.yaml
        with paths compatible with the Docker mount.
        """
        import yaml
        
        with open(original_yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Fix paths for Docker environment
        # We mount 'dataset_folder' to '/usr/src/dataset'
        config['path'] = '/usr/src/dataset'
        
        # Adjust train/val/test paths to be simple relative paths if they aren't already
        # We explicitly set them to what we observed in the directory structure
        # Assuming standard structure exists if not explicitly weird
        
        # Check if directories exist in the dataset folder, verify naming (val vs valid)
        # Based on user's listing: train, valid, test exist
        for key in ['train', 'val', 'test']:
            if key in config:
                # Naive fix: just set to folder name if it exists, otherwise keep original
                # But safer to just set 'train', 'valid', 'test' if they exist locally
                
                # Mapping common names
                if key == 'train':
                    if (dataset_folder / 'train').exists(): config['train'] = 'train/images'
                elif key == 'val':
                    if (dataset_folder / 'valid').exists(): config['val'] = 'valid/images'
                    elif (dataset_folder / 'val').exists(): config['val'] = 'val/images'
                elif key == 'test':
                    if (dataset_folder / 'test').exists(): config['test'] = 'test/images'

        docker_yaml_path = dataset_folder / "data_docker.yaml"
        with open(docker_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        return "data_docker.yaml"

    def start_training_container(self, model_name: str, epochs: int, batch_size: int, project_name: str, extra_args: dict = None):
        self._ensure_docker_client()
        if not self.client:
            raise RuntimeError("Docker client not initialized. Is Docker running?")

        dataset_folder, original_yaml_path = self._find_dataset_path()
        if not dataset_folder:
            raise FileNotFoundError(f"Could not find 'data.yaml' in {self.dataset_root_search}")

        # Create the docker specific config
        docker_config_name = self._create_docker_config(original_yaml_path, dataset_folder)

        # Prepare base command
        cmd = (
            f"yolo train "
            f"model={model_name}.pt "
            f"data=/usr/src/dataset/{docker_config_name} "
            f"epochs={epochs} "
            f"batch={batch_size} "
            f"project=/usr/src/runs "
            f"name={project_name}"
        )

        # Append all extra training arguments from config page
        if extra_args:
            for key, value in extra_args.items():
                # Convert Python bools to lowercase for YOLO CLI
                if isinstance(value, bool):
                    value = str(value).lower()
                cmd += f" {key}={value}"

        volumes = {
            str(dataset_folder): {'bind': '/usr/src/dataset', 'mode': 'rw'},
            str(self.runs_dir): {'bind': '/usr/src/runs', 'mode': 'rw'}
        }

        device_requests = [
            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
        ]

        try:
            container = self.client.containers.run(
                image="ultralytics/ultralytics:latest",
                command=cmd,
                volumes=volumes,
                device_requests=device_requests,
                detach=True,
                shm_size="8g"
            )
            return container.id
        except docker.errors.APIError as e:
            raise RuntimeError(f"Docker API Error: {e}")

    def get_container_status(self, container_id: str):
        self._ensure_docker_client()
        if not self.client:
            return "docker_connection_error"
            
        try:
            container = self.client.containers.get(container_id)
            return container.status
        except docker.errors.NotFound:
            return "not_found"

    def get_container_logs(self, container_id: str):
        self._ensure_docker_client()
        if not self.client:
            return "Docker connection error: Is Docker running?"
            
        try:
            container = self.client.containers.get(container_id)
            # return logs as string
            return container.logs()
        except docker.errors.NotFound:
            return ""

    def get_training_metrics(self, project_name: str):
        """
        Reads the results.csv from the run folder and returns data for all epochs.
        """
        results_path = self.runs_dir / project_name / "results.csv"
        if not results_path.exists():
            return []
            
        try:
            with open(results_path, 'r') as f:
                # YOLO CSV headers often have leading spaces
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    return []
                
                cleaned_rows = []
                for row in rows:
                     # Clean keys and values (remove extra whitespace)
                    cleaned_rows.append({k.strip(): v.strip() for k, v in row.items()})
                
                return cleaned_rows
        except Exception as e:
            print(f"Error reading metrics: {e}")
            return []
    
    def stop_training_container(self, container_id: str):
        self._ensure_docker_client()
        if not self.client:
            raise RuntimeError("Docker connection error: Is Docker running?")

        try:
            container = self.client.containers.get(container_id)
            container.stop()
        except docker.errors.NotFound:
            return "not_found"
        except docker.errors.APIError as e:
            raise RuntimeError(f"Docker API Error: {e}")
