"""Build a Russian report directly from the recorded experiment results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = {
    'mnist_comparison': 'Полносвязная сеть и CNN на MNIST',
    'cifar_comparison': 'Полносвязная сеть и CNN на CIFAR-10',
    'kernel': 'Размер ядра при одинаковом числе параметров',
    'depth': 'Глубина свёрточных сетей',
    'custom': 'Кастомные слои и стандартные аналоги',
    'residual': 'Варианты остаточных блоков',
}


def summarize(group, rows):
    """Format comparable metrics and locate the measured generalization gaps."""
    table = ['| Модель | Параметры | Train acc | Val acc | Test acc | Обучение, с | Инференс, мс/изобр. |',
             '|---|---:|---:|---:|---:|---:|---:|']
    gradients = ['| Модель | RMS первого веса | RMS последнего веса | Размах val loss, эпохи 3–5 |',
                 '|---|---:|---:|---:|']
    for name, row in rows.items():
        last = row['history'][-1]
        table.append(f"| {name} | {row['params']} | {last['train']['accuracy']:.4f} | "
                     f"{last['val']['accuracy']:.4f} | {row['test']['accuracy']:.4f} | "
                     f"{row['seconds']:.2f} | {1000 * row['inference'] / 2048:.4f} |")
        values = list(last['train']['gradients'].values())
        losses = [e['val']['loss'] for e in row['history'][-3:]]
        gradients.append(f'| {name} | {values[0]:.3e} | {values[-1]:.3e} | '
                         f'{max(losses) - min(losses):.4f} |')
    selected = max(rows, key=lambda name: rows[name]['history'][-1]['val']['accuracy'])
    last = rows[selected]['history'][-1]
    gap = last['train']['accuracy'] - last['val']['accuracy']
    links = [f'[Кривые обучения](homework/plots/{group}.png)',
             f'[Градиенты](homework/plots/{group}_gradients.png)',
             f'[Исходные метрики](homework/results/{group}/metrics.json)']
    if group == 'cifar_comparison':
        links.append(f'[Матрицы ошибок](homework/plots/{group}_confusion.png)')
    if group in ('kernel', 'depth'):
        links.extend(f'[{name}: карта признаков](homework/plots/{group}_{name}.png)'
                     for name in rows)
    details = f'По итоговой валидационной точности выбран вариант **{selected}**. Разрыв train–val составил {gap:+.4f}.'
    if group == 'kernel':
        rf = ', '.join(f"{name} — {row['receptive_field']}×{row['receptive_field']}"
                       for name, row in rows.items())
        details += f' Теоретические рецептивные поля после второго слоя и pooling: {rf}.'
    return '\n\n'.join(('\n'.join(table), details, '\n'.join(gradients), ' · '.join(links)))


def main():
    """Write a report only after each of the six groups has completed."""
    text = [
        '# Урок 4. Свёрточные сети',
        '## Задание и цель работы\n\nСравнить CNN и полносвязные сети, '
        'исследовать размер ядра, глубину, остаточные связи и кастомные слои.',
        '## Использованные методы\n\nAdam, шаг 0,001, батч 64, 5 эпох, CPU и seed 42. '
        'Для каждого набора использованы 4096 обучающих и 1024 валидационных изображения '
        'из официальной обучающей части и 2048 из официальной тестовой части. '
        'Тест использован после обучения, выбор вариантов выполнен по валидационной точности.',
        '## Реализация\n\nПростая CNN взята из примера преподавателя. '
        'Остаточные блоки реализованы с identity shortcut. Для кастомных слоёв '
        'операции PyTorch обеспечивают backward через autograd. Проверены аналитические '
        'свойства простых входов и численные производные. '
        'RMS градиента усредняется по батчам для каждой матрицы весов. '
        'Время инференса измеряется в eval после прогрева без загрузчика.',
        '## Эксперименты, результаты, графики и таблицы',
    ]
    results = {}
    for group, label in GROUPS.items():
        rows = json.loads((ROOT / 'results' / group / 'metrics.json').read_text(encoding='utf-8'))
        results[group] = rows
        text.append(f'### {label}\n\n{summarize(group, rows)}')
    def difference(group, left, right):
        return 100 * (results[group][left]['test']['accuracy']
                      - results[group][right]['test']['accuracy'])

    text.append('## Выводы\n\nСравнения относятся к фиксированным подвыборкам и пяти эпохам. '
                'При равном числе параметров ширина слоёв менялась вместе с размером ядра. '
                'Ранние и поздние градиенты оценивались через RMS, а динамика '
                'обобщения — через train–val разрыв и валидационные кривые.\n\n'
                f'На MNIST простая CNN опередила FC на {difference("mnist_comparison", "CNN", "FC"):.2f} п.п. '
                f'На CIFAR-10 остаточная CNN опередила FC на {difference("cifar_comparison", "Residual", "FC"):.2f} п.п. '
                f'Разница версии с dropout относительно обычной остаточной CNN составила '
                f'{difference("cifar_comparison", "Residual-dropout", "Residual"):+.2f} п.п.\n\n'
                f'Разница residual6 и обычной шестислойной CNN составила '
                f'{difference("depth", "residual6", "6"):+.2f} п.п. В residual6 также используется '
                'BatchNorm, поэтому сравнение отражает совместный эффект нормализации и остаточных связей.')
    (ROOT.parent / 'REPORT.md').write_text('\n\n'.join(text) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
