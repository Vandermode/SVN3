# SKSI
python -u test_cifar10.py --work-path ./experiments/cifar10/ablation/oenet_o3x1_t1 --no-log
# LKSI*
python -u test_cifar10.py --work-path ./experiments/cifar10/ablation/oenet_o15x1_t1 --no-log
# LKSI
python -u test_cifar10.py --work-path ./experiments/cifar10/ablation/oenet_o3x7_t1 --no-log
# SKSV*
python -u test_cifar10.py --work-path ./experiments/cifar10/ablation/oenet_o3x1sv0_t1 --no-log
# SKSV
python -u test_cifar10.py --work-path ./experiments/cifar10/ablation/oenet_o3x1sv_t1 --no-log
# LKSV
python -u test_cifar10.py --work-path ./experiments/cifar10/ablation/oenet_o3x7sv_t1 --no-log
