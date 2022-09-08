/**************************************************************************************************************
*   date : 20220904
*   task : 现在有一个从oracle数据库读数据的功能需求，要求输入地址、端口、数据库名、用户名、密码、表名，读取该表全部数据
*	OCCI : Oracle C++ Call Interface (OCCI)是Oracle自带的一套应用程序编程接口,它允许C++程序与一个或者多个Oracle数据库进行交互
**************************************************************************************************************/
/*
#include <iostream>
#define WIN32COMMON //避免函数重定义错误
#include <occi.h>
using namespace std;
using namespace oracle::occi;


int main()
{
	system("pause");
	//创建OCCI上下文环境
	Environment* env = Environment::createEnvironment();
	if (NULL == env) {
		printf("createEnvironment error.\n");
		return -1;
	}
	else
		cout << "success" << endl;

	string name = "SCOTT";
	string pass = "#123Qweasd";
	string srvName = "localhost:1521/orcl";

	try
	{
		//创建数据库连接
		Connection* conn = env->createConnection(name, pass, srvName);//用户名，密码，数据库名
		if (NULL == conn) {
			printf("createConnection error.\n");
			return -1;
		}
		else
			cout << "conn success" << endl;

		//	数据操作,创建Statement对象
		Statement* pStmt = NULL;    // Statement对象
		pStmt = conn->createStatement();
		if (NULL == pStmt) {
			printf("createStatement error.\n");
			return -1;
		}

		//--------查询---------  
		// 指定DML为自动提交  
		pStmt->setAutoCommit(TRUE);
		// 设置执行的SQL语句  
		pStmt->setSQL("SELECT * FROM emp");

		// 执行SQL语句  
		std::string strTemp;
		ResultSet* pRs = pStmt->executeQuery();
		while (pRs->next()) {
			strTemp = pRs->getString(3);
			printf("user:%s.\n", strTemp.c_str());
		}
		pStmt->closeResultSet(pRs);


		// 终止Statement对象  
		conn->terminateStatement(pStmt);

		//	关闭连接
		env->terminateConnection(conn);
		// pEnv->terminateConnection(pConn);  
	}
	catch (SQLException e)
	{
		cout << e.what() << endl;
		system("pause");
		return -1;
	}


	// 释放OCCI上下文环境  
	Environment::terminateEnvironment(env);
	cout << "end!" << endl;
	system("pause");
	return 0;
}
*/


#define WIN32COMMON
#include<cstdlib>
#include"occiemp.h"

using namespace std;
using namespace oracle::occi;


int main(void)
{	
	//OracleDB默认用户之一
	string username = "SCOTT";
	string password = "#123Qweasd";
	string srvName = "10.81.17.48:1521/orcl";

	try
	{
		occiemp* emp = new occiemp(username, password, srvName);
		emp->displayAllRows();

		//EMP info{"yyy","1234","yyyaa"};

		//emp->insertRow(info);

		//emp->displayAllRows();

		//emp->updateRow("ayyy","yyy");

		//emp->displayAllRows();

		//emp->deleteRow("yyy");

		//emp->displayAllRows();

		delete emp;
	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl;
		system("pause");
		return -1;
	}

	return 0;
}


/*
参考：
https://docs.oracle.com/en/database/oracle/oracle-database/21/lncpp/loe.html
https://blog.csdn.net/qq_40677159/article/details/118723779
http://blog.itpub.net/16203369/viewspace-1116537/
*/

