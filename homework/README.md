# Домашняя работа к уроку 4

[Задание](../HOMEWORK.md) · [Отчёт и графики](../REPORT.md)

Из корня репозитория выполните:

```powershell
python -m pip install torch torchvision matplotlib
python -m unittest -v
python -m homework.homework_cnn_vs_fc_comparison
python -m homework.homework_cnn_architecture_analysis
python -m homework.homework_custom_layers_experiments
python -m homework.build_report
```

По умолчанию torchvision хранит наборы в data/. Для использования уже загруженных MNIST и CIFAR-10 укажите путь в переменной LESSON_DATA_ROOT. JSON сохраняется в homework/results, графики — в homework/plots.

В матрицах ошибок CIFAR-10 номера 0–9 соответствуют классам airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
