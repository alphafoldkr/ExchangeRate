import subprocess

# 0.01 * b/256 
subprocess.call(['python', "tc_training.py","--device", "cuda","--threshold", "0.26", "--dimension", "64","--batch_size","512","--learning_rate","0.00001","--weight_decay","0.21","--dropout","0.3","--epochs","150"])
