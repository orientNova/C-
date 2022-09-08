#include "occiemp.h"
#include <iomanip> 

//构造函数
occiemp::occiemp(string user, string passwd, string db)
{
	try
	{
		this->env = Environment::createEnvironment();//创建环境
		this->conn = env->createConnection(user, passwd, db);//创建连接
	}
	catch (SQLException ex)
	{
		cout << "Error number: " << ex.getErrorCode() << endl;
		cout << ex.getMessage() << endl;
	}

}

//析构函数
occiemp::~occiemp()
{
	if (this->env != nullptr)
	{
		this->env->terminateConnection(this->conn);//释放连接
	}
	if (this->env)
	{
		Environment::terminateEnvironment(this->env);//释放环境
	}
}

//显示所有数据
void occiemp::displayAllRows()
{
	string sqlStmt = "select * from emp";//order by empno
	this->stmt = this->conn->createStatement(sqlStmt);
	//执行查询语句
	ResultSet* rset = this->stmt->executeQuery();//ResultSet提供对通过执行生成的数据表的访问Statement。表行按顺序检索。在一行中，可以按任何顺序访问列值。ResultSet保持光标指向其当前数据行。最初，光标位于第一行之前。next() 方法将光标移动到下一行。
	try
	{
		cout << "username" << setw(20) << "password" << setw(20) << "nickname" << endl;
		while (rset->next())
		{
			cout << rset->getString(1) << setw(20) << rset->getString(2) << setw(20) << rset->getString(3) << endl;
		}
	}
	catch (SQLException ex)
	{
		cout << "Exception thrown for displayAllRows" << endl;
		cout << "Error number: " << ex.getErrorCode() << endl;
		cout << ex.getMessage() << endl;
	}
	this->stmt->closeResultSet(rset);//释放集合数据
	this->conn->terminateStatement(this->stmt);//释放SQL语句
}

//更新数据
void occiemp::updateRow(string nickname, string username)
{
	string sqlStmt = "UPDATE JAVATEST1 SET \"nickname\" = :x WHERE \"username\" = :y ";
	try
	{
		this->stmt = this->conn->createStatement(sqlStmt);
		stmt->setString(1, nickname);
		stmt->setString(2, username);
		//执行非查询语句
		unsigned int res = stmt->executeUpdate();
		if (res > 0)
		{
			cout << "update - Success " << res << " 行受影响。" << endl;
		}

	}
	catch (SQLException ex)
	{
		cout << "Exception thrown for updateRow" << endl;
		cout << "Error number: " << ex.getErrorCode() << endl;
		cout << ex.getMessage() << endl;
	}

	this->conn->terminateStatement(this->stmt);
}

//根据条件删除数据
void occiemp::deleteRow(string username)
{
	string sqlStmt = "DELETE FROM JAVATEST1 WHERE \"username\" = :x ";

	try
	{
		this->stmt = this->conn->createStatement(sqlStmt);
		this->stmt->setString(1, username);
		unsigned int res = this->stmt->executeUpdate();
		if (res > 0)
		{
			cout << "delete - Success" << res << " 行受影响。" << endl;
		}

	}
	catch (SQLException ex)
	{
		cout << "Exception thrown for deleteRow" << endl;
		cout << "Error number: " << ex.getErrorCode() << endl;
		cout << ex.getMessage() << endl;
	}

	this->conn->terminateStatement(this->stmt);
}

//插入一行数据
void occiemp::insertRow(EMP emp)
{
	string sqlStmt = "INSERT INTO JAVATEST1 VALUES (:x1, :x2, :x3)";
	this->stmt = this->conn->createStatement(sqlStmt);
	try
	{
		this->stmt->setString(1, emp.username);
		this->stmt->setString(2, emp.password);
		this->stmt->setString(3, emp.nickname);

		unsigned int res = this->stmt->executeUpdate();
		if (res > 0)
		{
			cout << "Data saved successfully ," << res << " 行数据！" << endl;
		}
	}
	catch (SQLException ex)
	{
		cout << "Exception thrown for insertRow of emp" << endl;
		cout << "Error number: " << ex.getErrorCode() << endl;
		cout << ex.getMessage() << endl;
	}
	this->conn->terminateStatement(this->stmt);//释放SQL语句
}

Date occiemp::Todate(string strtime)
{
	try
	{
		int year = stoi((strtime.substr(0, 4)));
		unsigned int month = stoi((strtime.substr(4, 2)));
		unsigned int day = stoi((strtime.substr(6, 2)));
		unsigned int hour = stoi((strtime.substr(8, 2)));
		unsigned int minute = stoi((strtime.substr(10, 2)));
		unsigned int seconds = stoi((strtime.substr(12, 2)));
		Date date(this->env, year, month, day, hour, minute, seconds);
		return date;

	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl;
		return nullptr;
	}

}