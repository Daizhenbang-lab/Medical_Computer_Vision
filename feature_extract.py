import os
import glob
import argparse
from utils.embedding_generation import load_model, generate_umap, csv2h5ad, create_dataloader, extract_features
from sklearn.preprocessing import normalize
import torch


# def generate_embeddings(args):
#     print(torch.cuda.is_available())
#     files = glob.glob(os.path.join(args.save_path, '*'))
#     print('Found ', len(files), ' files')
#     print('------------------------------------------------------------------------------')
#     backbone = load_model(args.model_path, args.architecture, args.imagenet)
#     for i, file in enumerate(files):
#         print(f'Generating embeddings for {os.path.basename(file)} ...    ({i + 1}/{len(files)})')
#         generate_umap(backbone, os.path.join(args.save_path, os.path.basename(file)), args.num_workers,
#                       args.experiment_name, args.neighbors)
#         csv2h5ad(os.path.join(args.save_path, os.path.basename(file)), args.experiment_name)
#         print(f'Embeddings generated for {os.path.basename(file)}    ({i + 1}/{len(files)})')
#         print('------------------------------------------------------------------------------')


def generate_embeddings(args):

    print(torch.cuda.is_available())
    backbone = load_model(args.model_path, args.architecture, args.imagenet)
    backbone.eval()

    all_embeddings = []
    all_filenames = []
    sample_names = []

    for root, dirs, files in os.walk(args.save_path):

        if len(dirs) == 0:
            #if file.endswith('.jpg'):
                #patch_path = os.path.join(root, file)
            print(f"Processing {root}...")
                #print(patch_path)
            dataloader = create_dataloader(os.path.dirname(root), args.num_workers)
            embeddings, filenames = extract_features(backbone, dataloader)
            #print(filenames)
            all_embeddings.append(embeddings)
            all_filenames.extend(filenames)

            #scan_name = os.path.basename(os.path.dirname(root))
            sample_names.extend([s.split('__')[0] for s in filenames])

    all_embeddings = torch.cat(all_embeddings, 0)
    all_embeddings = normalize(all_embeddings)
    print(sample_names)
    generate_umap(args.save_path,
                  args.experiment_name, all_embeddings, all_filenames, sample_names)

    csv2h5ad(args.save_path, args.experiment_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Feature extraction configuration')
    parser.add_argument('--save_path', type=str, default='test_sample_56_overlap0.5/sample',
                        help='directory to save the extracted patches')
    parser.add_argument('--architecture', type=str, default='resnet18',
                        help='Model architecture (default: resnet18)')
    parser.add_argument('--model_path', type=str, default='checkpoints_/simclr-epoch=17-train_loss_ssl=4.29.ckpt',
                        help='Path to model weights')
    parser.add_argument('--imagenet', action='store_true')
    parser.add_argument('--no_imagenet', dest='imagenet', action='store_false')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--neighbors', type=int, default=2,
                        help='Number of n_neighbors in UMAP')
    parser.add_argument('--experiment_name', type=str, default='test_neighbor2',
                        help='Name of the experiment')

    args = parser.parse_args()
    print(args)
    generate_embeddings(args)
    print('Script finished!')