import sagemaker
from sagemaker.pytorch import PyTorch

# 1. Configure your SageMaker session
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::658450796965:role/service-role/AmazonSageMaker-ExecutionRole-20250705T225311"

# 2. Define the hyperparameters for your training job
# These will be passed as command-line arguments to your train.py script
hyperparameters = {

}

# 3. Create a PyTorch Estimator
estimator = PyTorch(
    entry_point='train.py',             # Your main training script
    source_dir='sm_train',            # The directory with your code and requirements.txt
    role=role,
    instance_count=1,                   # The number of machines to use for training
    instance_type='ml.g4dn.xlarge',      # The type of machine to use (a GPU instance is recommended)
    framework_version='2.0.0',          # The PyTorch version
    py_version='py310',                 # The Python version
    hyperparameters=hyperparameters
)

# 4. Launch the training job
estimator.fit()

print("Training job launched. You can monitor its progress in the SageMaker console.")