import os
import torch
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from map_action_data_loader import map_action_test_data_loader, map_action_data_loader
import m_a_detection_model as model
from training_utils import ModelTrainer

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

save_path = "/home/mapaction/mapaction_env/Map-Action-Model/model/MAISM2.pth"
num_classes = 3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with mlflow.start_run(experiment_id=1):
    Model =model.map_action_instance_segmentation_model(num_classes)


    loss_fn = torch.nn.CrossEntropyLoss()
    params = [p for p in Model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    print(len(map_action_test_data_loader))
    trainer = ModelTrainer(model=Model,
                            train_loader=map_action_data_loader,
                        test_loader= map_action_test_data_loader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        device=device)

    num_epochs = 10


    training_result = trainer.train(epochs=num_epochs)
torch.save(Model.state_dict(), save_path)
