# Applying-Grad_CAM-
In this repository I have applied Grad Cam Algorithm for visualizing the activation maps on CIFAR-10 dataset using Resnet18 Architecture

# Achieved accuracy of 88.77% in 16th epoch when optimizer used was SGD(Stochastic Gradient Descent)

# Achieved accuracy of 85.44% in 40th epoch when optimizer used was RMS_PROP

# Logs when used Optimizer SGD(Stochastic Gradient Descent)

 0%|          | 0/391 [00:00<?, ?it/s]True
Epoch: 1 Learning_Rate [0.0040000000000000036]
Loss=0.9655207395553589 Batch_id=390 Accuracy=51.53: 100%|██████████| 391/391 [00:53<00:00,  7.30it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0096, Accuracy: 5926/10000 (59.26%)

Epoch: 2 Learning_Rate [0.005944337266504118]
Loss=0.539883017539978 Batch_id=390 Accuracy=70.60: 100%|██████████| 391/391 [00:54<00:00,  7.24it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0066, Accuracy: 7238/10000 (72.38%)

Epoch: 3 Learning_Rate [0.011619830424103306]
Loss=0.7512584328651428 Batch_id=390 Accuracy=74.78: 100%|██████████| 391/391 [00:54<00:00,  7.13it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0058, Accuracy: 7520/10000 (75.20%)

Epoch: 4 Learning_Rate [0.020566684770626315]
Loss=0.670807957649231 Batch_id=390 Accuracy=77.82: 100%|██████████| 391/391 [00:55<00:00,  7.09it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0053, Accuracy: 7716/10000 (77.16%)

Epoch: 5 Learning_Rate [0.03206007937590945]
Loss=0.5162845849990845 Batch_id=390 Accuracy=80.88: 100%|██████████| 391/391 [00:55<00:00,  7.06it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0046, Accuracy: 8073/10000 (80.73%)

Epoch: 6 Learning_Rate [0.04516888776288231]
Loss=0.402390718460083 Batch_id=390 Accuracy=82.95: 100%|██████████| 391/391 [00:55<00:00,  7.04it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 8206/10000 (82.06%)

Epoch: 7 Learning_Rate [0.058831112237117685]
Loss=0.2946186363697052 Batch_id=390 Accuracy=85.60: 100%|██████████| 391/391 [00:55<00:00,  7.04it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0045, Accuracy: 8156/10000 (81.56%)

Epoch: 8 Learning_Rate [0.07193992062409055]
Loss=0.30213266611099243 Batch_id=390 Accuracy=87.22: 100%|██████████| 391/391 [00:55<00:00,  7.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0044, Accuracy: 8133/10000 (81.33%)

Epoch: 9 Learning_Rate [0.08343331522937368]
Loss=0.4079854488372803 Batch_id=390 Accuracy=88.42: 100%|██████████| 391/391 [00:55<00:00,  7.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0040, Accuracy: 8344/10000 (83.44%)

Epoch: 10 Learning_Rate [0.0923801695758967]
Loss=0.351257860660553 Batch_id=390 Accuracy=90.30: 100%|██████████| 391/391 [00:55<00:00,  7.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0038, Accuracy: 8479/10000 (84.79%)

Epoch: 11 Learning_Rate [0.09805566273349588]
Loss=0.10492327064275742 Batch_id=390 Accuracy=91.47: 100%|██████████| 391/391 [00:55<00:00,  7.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 8439/10000 (84.39%)

Epoch: 12 Learning_Rate [0.1]
Loss=0.18444597721099854 Batch_id=390 Accuracy=92.56: 100%|██████████| 391/391 [00:55<00:00,  7.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0040, Accuracy: 8474/10000 (84.74%)

Epoch: 13 Learning_Rate [0.09968561175222017]
Loss=0.28451329469680786 Batch_id=390 Accuracy=93.68: 100%|██████████| 391/391 [00:55<00:00,  7.06it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0038, Accuracy: 8627/10000 (86.27%)

Epoch: 14 Learning_Rate [0.09874640062350874]
Loss=0.17398294806480408 Batch_id=390 Accuracy=94.21: 100%|██████████| 391/391 [00:55<00:00,  7.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0039, Accuracy: 8623/10000 (86.23%)

Epoch: 15 Learning_Rate [0.09719417773875232]
Loss=0.1689492017030716 Batch_id=390 Accuracy=95.01: 100%|██████████| 391/391 [00:55<00:00,  7.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8784/10000 (87.84%)

Epoch: 16 Learning_Rate [0.09504846320134738]
Loss=0.1023731380701065 Batch_id=390 Accuracy=95.63: 100%|██████████| 391/391 [00:55<00:00,  7.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0033, Accuracy: 8877/10000 (88.77%)



# Logs when used Optimizer RMS_PROP

Epoch: 1 Learning_Rate [0.0040000000000000036]
Loss=1.7170934677124023 Batch_id=390 Accuracy=28.64: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.54it/s]

Test set: Average loss: 0.0134, Accuracy: 3674/10000 (36.74%)

Epoch: 2 Learning_Rate [0.005944337266504118]
Loss=1.4049631357192993 Batch_id=390 Accuracy=42.24: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.53it/s]

Test set: Average loss: 0.0122, Accuracy: 4471/10000 (44.71%)

Epoch: 3 Learning_Rate [0.011619830424103306]
Loss=1.0708088874816895 Batch_id=390 Accuracy=50.28: 100%|███████████████████████████| 391/391 [01:11<00:00,  7.59it/s]

Test set: Average loss: 0.0093, Accuracy: 5736/10000 (57.36%)

Epoch: 4 Learning_Rate [0.020566684770626315]
Loss=1.1294832229614258 Batch_id=390 Accuracy=56.05: 100%|███████████████████████████| 391/391 [01:11<00:00,  7.41it/s]

Test set: Average loss: 0.0432, Accuracy: 3093/10000 (30.93%)

Epoch: 5 Learning_Rate [0.03206007937590945]
Loss=1.2326918840408325 Batch_id=390 Accuracy=58.97: 100%|███████████████████████████| 391/391 [01:11<00:00,  7.15it/s]

Test set: Average loss: 0.0107, Accuracy: 5768/10000 (57.68%)

Epoch: 6 Learning_Rate [0.04516888776288231]
Loss=0.9509415626525879 Batch_id=390 Accuracy=59.94: 100%|███████████████████████████| 391/391 [01:12<00:00,  7.47it/s]

Test set: Average loss: 0.0078, Accuracy: 6495/10000 (64.95%)

Epoch: 7 Learning_Rate [0.058831112237117685]
Loss=0.9316840171813965 Batch_id=390 Accuracy=67.95: 100%|███████████████████████████| 391/391 [01:09<00:00,  7.62it/s]

Test set: Average loss: 0.0074, Accuracy: 6934/10000 (69.34%)

Epoch: 8 Learning_Rate [0.07193992062409055]
Loss=1.0655241012573242 Batch_id=390 Accuracy=69.30: 100%|███████████████████████████| 391/391 [01:09<00:00,  7.16it/s]

Test set: Average loss: 0.0084, Accuracy: 6409/10000 (64.09%)

Epoch: 9 Learning_Rate [0.08343331522937368]
Loss=118.21485900878906 Batch_id=390 Accuracy=67.72: 100%|███████████████████████████| 391/391 [01:09<00:00,  7.63it/s]

Test set: Average loss: 12.9510, Accuracy: 1224/10000 (12.24%)

Epoch: 10 Learning_Rate [0.0923801695758967]
Loss=2.277097225189209 Batch_id=390 Accuracy=13.15: 100%|████████████████████████████| 391/391 [01:10<00:00,  7.43it/s]

Test set: Average loss: 0.0706, Accuracy: 1112/10000 (11.12%)

Epoch: 11 Learning_Rate [0.09805566273349588]
Loss=2.263171434402466 Batch_id=390 Accuracy=16.54: 100%|████████████████████████████| 391/391 [01:10<00:00,  7.47it/s]

Test set: Average loss: 0.0159, Accuracy: 1955/10000 (19.55%)

Epoch: 12 Learning_Rate [0.1]
Loss=1.7858902215957642 Batch_id=390 Accuracy=26.22: 100%|███████████████████████████| 391/391 [01:12<00:00,  7.65it/s]

Test set: Average loss: 0.0131, Accuracy: 3395/10000 (33.95%)

Epoch: 13 Learning_Rate [0.09968561175222017]
Loss=1.3260982036590576 Batch_id=390 Accuracy=42.08: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.54it/s]

Test set: Average loss: 0.0122, Accuracy: 4659/10000 (46.59%)

Epoch: 14 Learning_Rate [0.09874640062350874]
Loss=1.4087852239608765 Batch_id=390 Accuracy=54.79: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.57it/s]

Test set: Average loss: 0.0099, Accuracy: 5642/10000 (56.42%)

Epoch: 15 Learning_Rate [0.09719417773875232]
Loss=0.9782121777534485 Batch_id=390 Accuracy=60.18: 100%|███████████████████████████| 391/391 [01:11<00:00,  7.36it/s]

Test set: Average loss: 0.0097, Accuracy: 5869/10000 (58.69%)

Epoch: 16 Learning_Rate [0.09504846320134738]
Loss=1.1018235683441162 Batch_id=390 Accuracy=62.80: 100%|███████████████████████████| 391/391 [01:15<00:00,  7.64it/s]

Test set: Average loss: 0.0089, Accuracy: 6173/10000 (61.73%)

Epoch: 17 Learning_Rate [0.09233624061657436]
Loss=0.9569516181945801 Batch_id=390 Accuracy=64.99: 100%|███████████████████████████| 391/391 [01:09<00:00,  5.62it/s]

Test set: Average loss: 0.0079, Accuracy: 6643/10000 (66.43%)

Epoch: 18 Learning_Rate [0.089091617757105]
Loss=0.9365575909614563 Batch_id=390 Accuracy=70.86: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.64it/s]

Test set: Average loss: 0.0072, Accuracy: 6934/10000 (69.34%)

Epoch: 19 Learning_Rate [0.08535539763797113]
Loss=1.1282166242599487 Batch_id=390 Accuracy=71.99: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.62it/s]

Test set: Average loss: 0.0063, Accuracy: 7530/10000 (75.30%)

Epoch: 20 Learning_Rate [0.0811745653949763]
Loss=1.091168761253357 Batch_id=390 Accuracy=75.16: 100%|████████████████████████████| 391/391 [01:10<00:00,  7.44it/s]

Test set: Average loss: 0.0056, Accuracy: 7612/10000 (76.12%)

Epoch: 21 Learning_Rate [0.07660169741935154]
Loss=0.5125759243965149 Batch_id=390 Accuracy=77.94: 100%|███████████████████████████| 391/391 [01:15<00:00,  5.20it/s]

Test set: Average loss: 0.0068, Accuracy: 7487/10000 (74.87%)

Epoch: 22 Learning_Rate [0.07169430017913009]
Loss=0.7758492827415466 Batch_id=390 Accuracy=79.24: 100%|███████████████████████████| 391/391 [01:15<00:00,  7.30it/s]

Test set: Average loss: 0.0062, Accuracy: 7503/10000 (75.03%)

Epoch: 23 Learning_Rate [0.06651408704194597]
Loss=0.3797483742237091 Batch_id=390 Accuracy=79.26: 100%|███████████████████████████| 391/391 [01:11<00:00,  7.21it/s]

Test set: Average loss: 0.0061, Accuracy: 7688/10000 (76.88%)

Epoch: 24 Learning_Rate [0.06112620219362893]
Loss=0.8175967931747437 Batch_id=390 Accuracy=81.63: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.44it/s]

Test set: Average loss: 0.0055, Accuracy: 7833/10000 (78.33%)

Epoch: 25 Learning_Rate [0.055598401412270175]
Loss=0.5639023780822754 Batch_id=390 Accuracy=82.48: 100%|███████████████████████████| 391/391 [01:12<00:00,  7.55it/s]

Test set: Average loss: 0.0063, Accuracy: 7666/10000 (76.66%)

Epoch: 26 Learning_Rate [0.0500002]
Loss=0.7224020957946777 Batch_id=390 Accuracy=83.51: 100%|███████████████████████████| 391/391 [01:11<00:00,  7.51it/s]

Test set: Average loss: 0.0052, Accuracy: 8084/10000 (80.84%)

Epoch: 27 Learning_Rate [0.04440199858772983]
Loss=0.7180603742599487 Batch_id=390 Accuracy=84.39: 100%|███████████████████████████| 391/391 [01:15<00:00,  7.54it/s]

Test set: Average loss: 0.0050, Accuracy: 8153/10000 (81.53%)

Epoch: 28 Learning_Rate [0.03887419780637107]
Loss=0.41364842653274536 Batch_id=390 Accuracy=84.71: 100%|██████████████████████████| 391/391 [01:10<00:00,  7.21it/s]

Test set: Average loss: 0.0050, Accuracy: 8229/10000 (82.29%)

Epoch: 29 Learning_Rate [0.03348631295805405]
Loss=0.602688193321228 Batch_id=390 Accuracy=86.53: 100%|████████████████████████████| 391/391 [01:11<00:00,  7.60it/s]

Test set: Average loss: 0.0055, Accuracy: 8082/10000 (80.82%)

Epoch: 30 Learning_Rate [0.028306099820869922]
Loss=0.5465839505195618 Batch_id=390 Accuracy=86.82: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.48it/s]

Test set: Average loss: 0.0048, Accuracy: 8274/10000 (82.74%)

Epoch: 31 Learning_Rate [0.023398702580648485]
Loss=0.32694312930107117 Batch_id=390 Accuracy=87.83: 100%|██████████████████████████| 391/391 [01:11<00:00,  7.51it/s]

Test set: Average loss: 0.0043, Accuracy: 8348/10000 (83.48%)

Epoch: 32 Learning_Rate [0.0188258346050237]
Loss=0.1724962294101715 Batch_id=390 Accuracy=88.90: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.53it/s]

Test set: Average loss: 0.0048, Accuracy: 8337/10000 (83.37%)

Epoch: 33 Learning_Rate [0.014645002362028864]
Loss=0.4132879376411438 Batch_id=390 Accuracy=89.80: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.53it/s]

Test set: Average loss: 0.0044, Accuracy: 8392/10000 (83.92%)

Epoch: 34 Learning_Rate [0.010908782242895008]
Loss=0.3135248124599457 Batch_id=390 Accuracy=90.87: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.39it/s]

Test set: Average loss: 0.0045, Accuracy: 8358/10000 (83.58%)

Epoch: 35 Learning_Rate [0.0076641593834256404]
Loss=0.117383673787117 Batch_id=390 Accuracy=91.43: 100%|████████████████████████████| 391/391 [01:10<00:00,  7.64it/s]

Test set: Average loss: 0.0044, Accuracy: 8409/10000 (84.09%)

Epoch: 36 Learning_Rate [0.004951936798652629]
Loss=0.2545124590396881 Batch_id=390 Accuracy=92.04: 100%|███████████████████████████| 391/391 [01:09<00:00,  7.66it/s]

Test set: Average loss: 0.0044, Accuracy: 8497/10000 (84.97%)

Epoch: 37 Learning_Rate [0.002806222261247683]
Loss=0.27505218982696533 Batch_id=390 Accuracy=92.84: 100%|██████████████████████████| 391/391 [01:09<00:00,  7.61it/s]

Test set: Average loss: 0.0043, Accuracy: 8512/10000 (85.12%)

Epoch: 38 Learning_Rate [0.0012539993764912555]
Loss=0.13705667853355408 Batch_id=390 Accuracy=93.40: 100%|██████████████████████████| 391/391 [01:15<00:00,  5.20it/s]

Test set: Average loss: 0.0044, Accuracy: 8513/10000 (85.13%)

Epoch: 39 Learning_Rate [0.0003147882477798485]
Loss=0.26092809438705444 Batch_id=390 Accuracy=93.66: 100%|██████████████████████████| 391/391 [01:15<00:00,  7.32it/s]

Test set: Average loss: 0.0043, Accuracy: 8533/10000 (85.33%)

Epoch: 40 Learning_Rate [4e-07]
Loss=0.1440194547176361 Batch_id=390 Accuracy=93.66: 100%|███████████████████████████| 391/391 [01:10<00:00,  7.63it/s]

Test set: Average loss: 0.0043, Accuracy: 8544/10000 (85.44%)
