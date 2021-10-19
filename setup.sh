echo "Setting up environment for Table Detection"
echo "Installing Dependecies for Table Detection"
python -m pip install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
pip install -r requirements.txt
echo "Downloading Table Detection Models"
gdown --id 11FgFTy0MyVUMGd00T_InEDaarB4qAlP8 -O content/
gdown --id 1WBk6kHHyvyEzoPBsRr2BvFY51zURjd4R -O content/
gdown --id 1PfA2uws919gc893-x9uMIz06zWEko8nF -O content/
echo "Setup Complete. Please run below command to start execution of server."
echo "steamlit run app.py"