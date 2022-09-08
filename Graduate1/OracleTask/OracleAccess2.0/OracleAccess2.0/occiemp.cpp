#include "occiemp.h"
#include <iomanip> 

//���캯��
occiemp::occiemp(string user, string passwd, string db)
{
	try
	{
		this->env = Environment::createEnvironment();//��������
		this->conn = env->createConnection(user, passwd, db);//��������
	}
	catch (SQLException ex)
	{
		cout << "Error number: " << ex.getErrorCode() << endl;
		cout << ex.getMessage() << endl;
	}

}

//��������
occiemp::~occiemp()
{
	if (this->env != nullptr)
	{
		this->env->terminateConnection(this->conn);//�ͷ�����
	}
	if (this->env)
	{
		Environment::terminateEnvironment(this->env);//�ͷŻ���
	}
}

//��ʾ��������
void occiemp::displayAllRows()
{
	string sqlStmt = "select * from emp";//order by empno
	this->stmt = this->conn->createStatement(sqlStmt);
	//ִ�в�ѯ���
	ResultSet* rset = this->stmt->executeQuery();//ResultSet�ṩ��ͨ��ִ�����ɵ����ݱ�ķ���Statement�����а�˳���������һ���У����԰��κ�˳�������ֵ��ResultSet���ֹ��ָ���䵱ǰ�����С���������λ�ڵ�һ��֮ǰ��next() ����������ƶ�����һ�С�
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
	this->stmt->closeResultSet(rset);//�ͷż�������
	this->conn->terminateStatement(this->stmt);//�ͷ�SQL���
}

//��������
void occiemp::updateRow(string nickname, string username)
{
	string sqlStmt = "UPDATE JAVATEST1 SET \"nickname\" = :x WHERE \"username\" = :y ";
	try
	{
		this->stmt = this->conn->createStatement(sqlStmt);
		stmt->setString(1, nickname);
		stmt->setString(2, username);
		//ִ�зǲ�ѯ���
		unsigned int res = stmt->executeUpdate();
		if (res > 0)
		{
			cout << "update - Success " << res << " ����Ӱ�졣" << endl;
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

//��������ɾ������
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
			cout << "delete - Success" << res << " ����Ӱ�졣" << endl;
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

//����һ������
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
			cout << "Data saved successfully ," << res << " �����ݣ�" << endl;
		}
	}
	catch (SQLException ex)
	{
		cout << "Exception thrown for insertRow of emp" << endl;
		cout << "Error number: " << ex.getErrorCode() << endl;
		cout << ex.getMessage() << endl;
	}
	this->conn->terminateStatement(this->stmt);//�ͷ�SQL���
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