#ifndef UCT_H
#define UCT_H
#include "Point.h"
#include "Judge.h"
#include "Strategy.h"
#include <math.h>
#include <iostream>

using namespace std;

#define C 0.5
#define PLAYER_TURN -1.0
#define AI_TURN 1.0
#define PLAYER_WIN -1.0
#define AI_WIN 1.0
#define TIED 0.0
#define UNFINISHED 10.0
#define ROUND_LIMIT 300000


class Node {
	int** board;//������Ϣ
	int* top;//ʣ�����ӵ�
	int row, colume;
	int x, y;//��һ������
	int noX, noY;
	double turn;//�ִ�
	double visitsum;
	int depth;
	double profit;
	Node* father;
	Node** children;
	int childrensum;//�ӽڵ����
	int* expandindex;
	int expandsum;//����չ�ڵ����

	friend class UCT;
	bool isEnd();
	int** getboard();
	int* gettop();
public:
	Node(int** _board,int* _top,int _row,int _colume,int _x,int _y,int _noX,int _noY,double _turn,int _depth,Node* _father):board(_board),top(_top),row(_row),colume(_colume),x(_x),y(_y),noX(_noX),noY(_noY),turn(_turn),depth(_depth),visitsum(0),profit(0.0),father(_father),childrensum(0),expandsum(0){
		children = new Node*[colume];
		expandindex = new int[colume];
		for(int i = 0; i < colume; i++) {
			if(top[i] > 0)
				expandindex[expandsum++] = i;
			children[i] = NULL;
		}
	}
	~Node(){
		for(int i = 0; i < row; i++)
			delete[] board[i];
		delete[] board;
		delete[] top;
		delete[] expandindex;
		for(int i = 0; i < childrensum; i++)
			delete children[i];
		delete[] children;
	}	
};

class UCT {
	Node* root;
	int row, colume;
	int noX, noY;

	Node* TreePollicy(Node* _root);
	double DefaultPolicy(Node* node);
	void BackUp(Node* node, double profit);
	Node* Expand(Node* _root);
	Node* BestChild(Node* _root,double c);
	double Profit(int** _board, int* top, int _x, int _y, double turn);
	void move(int** _board, int* _top, double tuen, int &x, int &y);//������Ӳ���
public:
	UCT(int _row, int _colume, int _noX, int _noY) : row(_row), colume(_colume), noX(_noX), noY(_noY) {}
	Point* search(int** _board, int* _top, int _lastX,int _lastY);
	~UCT() {
		delete root;
	}
};
#endif // !UCT_H
