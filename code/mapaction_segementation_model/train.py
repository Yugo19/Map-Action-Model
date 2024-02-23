import torch
from data_loader import mapaction_test_data_loader, mapction_data_loader
import model
from training_pipeline import ModelTrainer

save_path = "/home/mapaction/mapaction_env/Map-Action-Model/model/test4.pth"
num_classes = 2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

Model =model.mapaction_instance_segmentation_model(num_classes)


loss_fn = torch.nn.CrossEntropyLoss()
params = [p for p in Model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

trainer = ModelTrainer(Model,
                     train_loader=mapction_data_loader,
                test_loader= mapaction_test_data_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device)


num_epochs = 5

training_result = trainer.train(epochs=num_epochs)
torch.save(Model.state_dict(), save_path)