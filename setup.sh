
# apt-get install -y supervisor

# mkdir /etc/supervisor/conf.d/
# cp tf.conf /etc/supervisor/conf.d/tensorboard.conf
pip install tensorboardX
pip install tensorflow
pip install git+https://github.com/lanpa/tensorboard-pytorch

# tensordboard logdir /output/runs
