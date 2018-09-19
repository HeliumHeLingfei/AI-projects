#include "UCT.h"

using namespace std;

bool Node::isEnd() {
	if(x == -1 && y == -1)
		return false;
	else if(turn == PLAYER_TURN && machineWin(x, y, row, colume, board))
		return true;
	else if(turn == AI_TURN && userWin(x, y, row, colume, board))
		return true;
	else if(isTie(colume, top))
		return true;
	else
		return false;
}

int** Node::getboard() {
	int** b = new int*[row];
	for(int i = 0; i < row; i++) {
		b[i] = new int[colume];
		for(int j = 0; j < colume; j++)
			b[i][j] = board[i][j];
	}
	return b;
}

int* Node::gettop() {
	int* t = new int[colume];
	for(int i = 0; i < colume; i++)
		t[i] = top[i];
	return t;
}

Node* UCT::TreePollicy(Node* _root) {
	while(!_root->isEnd()) {
		if(_root->expandsum > 0)
			return Expand(_root);
		else {
			_root = BestChild(_root, C);
		}
	}
	return _root;
}

Node* UCT::Expand(Node* _root) {
	int index = rand() % _root->expandsum;
	int** b = _root->getboard();
	int*t = _root->gettop();
	int _y = _root->expandindex[index];
	int _x = --t[_y];
	b[_x][_y] = (_root->turn == PLAYER_TURN) ? 1 : 2;
	if(_x - 1 == noX && _y == noY)
		t[_y]--;
	_root->children[_root->childrensum++] = new Node(b,t,row,colume,_x,_y,noX,noY,-_root->turn,_root->depth+1,_root);
	_root->expandindex[index] = _root->expandindex[--_root->expandsum];
	return _root->children[_root->childrensum - 1];
}

Node* UCT::BestChild(Node* _root,double c) {
	double max_profit = -INT_MAX;
	Node* node;
	for(int i = 0; i < _root->childrensum; i++) {
		double vsum = _root->children[i]->visitsum;
		double tem = _root->turn *  _root->children[i]->profit;
		double profit = tem / vsum + c * sqrt(2.0 * log(_root->visitsum) / vsum);
		if(profit > max_profit || profit == max_profit && rand() % 2 == 0) {
			max_profit = profit;
			node = _root->children[i];
		}
	}
	return node;
}

double UCT::Profit(int** _board, int* top, int _x, int _y, double turn) {
	if(turn == PLAYER_TURN && userWin(_x, _y, row, colume, _board))
		return PLAYER_WIN;
	else if(turn == AI_TURN && machineWin(_x, _y, row, colume, _board))
		return AI_WIN;
	else if(isTie(colume, top))
		return TIED;
	else
		return UNFINISHED;
}

void UCT::move(int** _board, int* _top, double turn, int &_x, int &_y) {
	do {
		_y = rand() % colume;
	} while(_top[_y]==0);
	_x = --_top[_y];
	_board[_x][_y] = (turn == PLAYER_TURN) ? 1 : 2;
	if(_x - 1 == noX && _y == noY)
		_top[_y]--;
}

double UCT::DefaultPolicy(Node* node) {
	int** board = node->getboard();
	int* top = node->gettop();
	double turn = node->turn;
	int x = node->x, y = node->y;
	double profit = Profit(board, top, x, y, -turn);
	while(profit == UNFINISHED) {
		move(board, top, turn, x, y);
		profit = Profit(board, top, x, y, turn);
		turn *= -1.0;
	}
	for(int i = 0; i < row; i++) 
		delete[] board[i];
	delete[] board;
	delete[] top;
	return profit;
}

void UCT::BackUp(Node* node, double profit) {
	Node* tem = node;
	while(tem) {
		tem->profit += profit;
		tem->visitsum++;
		tem = tem->father;
	}
}

Point* UCT::search(int** _board, int* _top, int _lastX, int _lastY) {
	int** board = new int*[row];
	for(int i = 0; i < row; i++) {
		board[i] = new int[colume];
		for(int j = 0; j < colume; j++)
			board[i][j] = _board[i][j];
	}
	int* top = new int[colume];
	for(int i = 0; i < colume; i++)
		top[i] = _top[i];
	root = new Node(board,top,row,colume,_lastX,_lastY,noX,noY,AI_TURN,0,NULL);
	int sum = 0;
	while(sum<ROUND_LIMIT) {
		sum++;
		Node* node = TreePollicy(root);
		double delta = DefaultPolicy(node);
		BackUp(node, delta);
	}
	Node* bestchild = BestChild(root, 0);
	Point* p=new Point(bestchild->x,bestchild->y);
	return p;
}

