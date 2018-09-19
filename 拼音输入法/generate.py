# -*- coding: UTF-8 -*-

from src.pypinyin import lazy_pinyin
import json
import re
from src.zhon.hanzi import punctuation
from math import log10
import numpy as np


# nue->nve lue->lve

def init():
	emission = {}
	# 预处理矩阵
	with open("./data/拼音汉字表.txt", 'r') as f:
		lines = f.readlines()
		for line in lines:
			arg = line.split()
			for character in arg[1:]:
				if character not in emission:
					emission[character] = {}
				emission[character][arg[0]] = 0.0

	with open("./data/characters.json", 'r') as f:
		characters = json.load(f)
	transitions = {}
	start = {}
	linestart = {}
	for ch0 in characters:
		transitions[ch0] = {}
		start[ch0] = 0.0
		linestart[ch0] = 1.0
		for ch1 in characters:
			transitions[ch0][ch1] = 0.0
	overcountpi = 0.0
	# 读入文件
	for months in range(1, 12):
		if months < 10:
			filename = "2016-0" + str(months) + ".txt"
		else:
			filename = "2016-" + str(months) + ".txt"
		with open("./sina_news_gbk/" + filename, 'r') as f:
			for (lineno, content) in enumerate(f):
				data = json.loads(content)
				# 预处理语句
				line = re.split(u"[%s]+" % (punctuation + ' '),
				                re.sub(r'\(.*?\)', '', re.sub(r'（.*?）', '', data['html'].replace("\n", '。'))))
				for t in line:
					l = re.sub('[^%s]+' % characters, '', t)
					if not l:
						continue
					if t[0] in characters:  #猜测由于这一句导致复杂度飙升
						linestart[t[0]] += 1.0
					lpy = lazy_pinyin(l)
					start[l[0]] += 1.0
					ch0 = l[0]
					if lpy[0] in emission[l[0]]:
						emission[l[0]][lpy[0]] += 1.0
					for index in range(1, len(l)):
						start[l[index]] += 1
						transitions[ch0][l[index]] += 1.0
						ch0 = l[index]
						if lpy[index] not in emission[l[index]]:
							continue
						emission[l[index]][lpy[index]] += 1.0
	sum = 0.0 - overcountpi
	pi = []
	a = []
	acount = {}
	bcount = {}
	pycount = {}
	pysum = {}
	py2ch = {}
	asum = 0.0
	bsum = 0.0
	# 计数
	with open("./data/拼音汉字表.txt", 'r') as f:
		lines = f.readlines()
		for line in lines:
			arg = line.split()
			pycount[arg[0]] = 0.0
			pysum[arg[0]] = len(arg) - 1
			py2ch[arg[0]] = arg[1:]
	for i in characters:
		acount[i] = [0.0, 0.0]
	for i in characters:
		bcount[i] = 0.0
		for py in emission[i]:
			if emission[i][py] > 0:
				bcount[i] += 1.0
				pycount[py] += 1.0
				bsum += 1.0
		for j in characters:
			if transitions[i][j] > 0:
				acount[i][0] += 1.0
				acount[j][1] += 1.0
				asum += 1.0
	# 平滑
	for i in characters:
		pi.append(start[i] + linestart[i])
		sum += start[i] + linestart[i]
		ai = []
		if start[i] != 0:
			for py in emission[i]:
				P = max(emission[i][py] - 0.75, 0.000027) + 0.75 * bcount[i] * pycount[py] / bsum
				emission[i][py] = log10(P / start[i])
			for j in characters:
				P = max(transitions[i][j] - 0.75, 0.000027) + 0.75 * acount[i][0] * acount[j][1] / asum
				ai.append(log10(P / start[i]))
		else:
			for py in emission[i]:
				emission[i][py] = log10(1 / pysum[py])
			for j in characters:
				ai.append(log10(1 / 6763))
		a.append(ai)
	for i in range(len(pi)):
		pi[i] = log10(pi[i] / sum)
	pi = np.array(pi)
	a = np.array(a)

	with open("./data/emissions.json", 'w') as f:
		json.dump(emission, f)
	with open("./data/py2ch.json", 'w') as f:
		json.dump(py2ch, f)
	a.dump("./data/transitions")
	pi.dump("./data/pi")


if __name__ == '__main__':
	init()
