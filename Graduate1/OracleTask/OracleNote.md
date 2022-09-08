# Oracle

## 0 任务描述

从oracle数据库读数据，要求输入地址、端口、数据库名、用户名、密码、表名，读取该表全部数据

## 1 OCCI

Oracle C++ Call Interface (OCCI) is an Application Programming Interface (API) that provides C++ applications access to data in an Oracle database. OCCI enables C++ programmers to use the full range of Oracle database operations, including SQL statement processing and object manipulation.

## 2 VS环境搭建

[VS C++ 使用 OCCI 连接调用 oracle](https://blog.csdn.net/weixin_41049188/article/details/106606192) 

## 3 创建本地数据库

[Oracle 11g下载及安装](https://blog.csdn.net/qq_34602804/article/details/114239776?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-114239776-blog-125398702.pc_relevant_vip_default&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-114239776-blog-125398702.pc_relevant_vip_default&utm_relevant_index=1) 

[ORA-12541: TNS: 无监听程序 的解决办法](https://blog.csdn.net/qq_34621658/article/details/98939526?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-98939526-blog-88412383.pc_relevant_vip_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-98939526-blog-88412383.pc_relevant_vip_default&utm_relevant_index=2) 

## 4 关键代码

其中标注//>>>[XXX]的即为所需接口

~~~C++
/*******************************************************************************************
*   date : 20220904
*   task : 现在有一个从oracle数据库读数据的功能需求，要求输入地址、端口、数据库名、用户名、密码、表名，读取该表全部数据
*******************************************************************************************/

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

	string name = "scott";//>>>[用户名]
	string pass = "tiger";//>>>[密码]
	string srvName = "localhost:1521/orcl";//>>>[输入地址、端口、数据库名]

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
		pStmt->setSQL("SELECT * FROM emp");//>>>[表名emp]

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

~~~

