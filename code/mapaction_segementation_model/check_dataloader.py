from data_loader import mapction_data_loader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


for imgs, targets in data_loader:
    imgs = list(img.to(device) for img in imgs)
    targets = [{k: v for k, v in t.items()} for t in targets]
    print(targets)