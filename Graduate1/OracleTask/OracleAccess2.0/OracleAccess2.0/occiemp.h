#ifndef __OCCIEMP_H
#define __OCCIEMP_H

#include <iostream>
#include <occi.h>
using namespace oracle::occi;
using namespace std;

typedef struct emptable
{
	string username;
	string password;
	string nickname;
}EMP;

//Ա����
class occiemp
{
public:
	occiemp(string user, string passwd, string db);
	~occiemp();

	void displayAllRows();//��ʾ��������
	void updateRow(string nickname, string username);
	void deleteRow(string nickname);
	void insertRow(EMP emp);//����һ������
	Date Todate(string time);
private:
	Environment* env = nullptr;//�����Ļ���
	Connection* conn = nullptr;//���ݿ����Ӿ��
	Statement* stmt = nullptr;//ָ��SQL���������

};

#endif // !__OCCIEMP_H

