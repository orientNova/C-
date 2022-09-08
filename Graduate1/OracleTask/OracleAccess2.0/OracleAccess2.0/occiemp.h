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

//员工表
class occiemp
{
public:
	occiemp(string user, string passwd, string db);
	~occiemp();

	void displayAllRows();//显示所有数据
	void updateRow(string nickname, string username);
	void deleteRow(string nickname);
	void insertRow(EMP emp);//插入一行数据
	Date Todate(string time);
private:
	Environment* env = nullptr;//上下文环境
	Connection* conn = nullptr;//数据库连接句柄
	Statement* stmt = nullptr;//指向SQL语句声明类

};

#endif // !__OCCIEMP_H

