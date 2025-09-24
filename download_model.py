from modelscope import HubApi
from modelscope import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--id_token', type=str, default='')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--folder_path', type=str, default='')
args = parser.parse_args()

# Log In 
api=HubApi()
api.login(args.id_token)

# download your model, the model_path is downloaded model path.
model_path = snapshot_download(model_id=args.model_name, 
        cache_dir=args.folder_path, local_dir=args.folder_path,)
