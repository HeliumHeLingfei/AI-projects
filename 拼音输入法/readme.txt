基于二阶隐马尔可夫模型的拼音输入法

本程序基于python3.6，除了python的原生模块还使用了pypinyin0.30.0，zhon1.1.5，numpy1.14.2三个库文件，其中前两个已放到src文件夹中。

generate.py读取预料并生成模型，生成了的模型储存在data文件夹中。共有characters.json, emission.json, py2ch.json, pi, transitions 五个文件。

viterbi.py加载模型，执行解码运算得出结果。
使用python viterbi.py，读取data文件夹中的input.txt，并按行依次输出解码出的中文，写入data\output.txt文件中。
使用python viterbi.py type，手动输入拼音，并且实时输出结果，输入quit结束。
使用python viterbi.py input_filename output_filename，则读取input文件并输出到output文件中。文件名。