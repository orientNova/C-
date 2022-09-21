#include "widget.h"
#include "./ui_widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    //清空
    CalcContent.clear();
    CalcTemp.clear();

    //绑定按键与处理函数
    connect(ui->pushButton_0, &QPushButton::clicked,[=](){btn_logic(0," ");});
    connect(ui->pushButton_1, &QPushButton::clicked,[=](){btn_logic(1," ");});
    connect(ui->pushButton_2, &QPushButton::clicked,[=](){btn_logic(2," ");});
    connect(ui->pushButton_3, &QPushButton::clicked,[=](){btn_logic(3," ");});
    connect(ui->pushButton_4, &QPushButton::clicked,[=](){btn_logic(4," ");});
    connect(ui->pushButton_5, &QPushButton::clicked,[=](){btn_logic(5," ");});
    connect(ui->pushButton_6, &QPushButton::clicked,[=](){btn_logic(6," ");});
    connect(ui->pushButton_7, &QPushButton::clicked,[=](){btn_logic(7," ");});
    connect(ui->pushButton_8, &QPushButton::clicked,[=](){btn_logic(8," ");});
    connect(ui->pushButton_9, &QPushButton::clicked,[=](){btn_logic(9," ");});

    connect(ui->pushButton_plus, &QPushButton::clicked,[=](){btn_logic(0,"+");});
    connect(ui->pushButton_minus, &QPushButton::clicked,[=](){btn_logic(0,"-");});
    connect(ui->pushButton_multiply, &QPushButton::clicked,[=](){btn_logic(0,"*");});
    connect(ui->pushButton_divide, &QPushButton::clicked,[=](){btn_logic(0,"/");});
    connect(ui->pushButton_calc, &QPushButton::clicked,[=](){btn_logic(0,"=");});
    connect(ui->pushButton_clear, &QPushButton::clicked,[=](){ CalcContent.clear();CalcTemp.clear();ui->lineEdit->setText(CalcContent); });
    connect(ui->pushButton_back, &QPushButton::clicked,[=](){CalcContent.chop(1);ui->lineEdit->setText(CalcContent);});
}

/**********************************************
*按键处理函数
*参数x: 按键值
*参数i: 1:数字 0:运算符
**********************************************/
void Widget::btn_logic(int x, QString i)
{
    if(i == " ")//输入数字
    {
        CalcContent += QString::number(x);
    }

    else//输入运算符
    {
        if(i != "=")
        {
            CalcTemp.insert(0,CalcContent);//将输入内容储存到CalcTemp[0]中
            CalcContent.clear();
            CalcTemp.insert(1,i);//将输入内容储存到CalcTemp[1]中
        }
        else
        {
            CalcTemp.insert(2,CalcContent);//将输入内容储存到CalcTemp[2]中
            CalcContent.clear();
            if(CalcTemp.at(1) == "+")
            {
                CalcContent = QString::number( QString(CalcTemp.at(0)).toInt() + QString(CalcTemp.at(2)).toInt() );
            }
            else if(CalcTemp.at(1) == "-")
            {
                CalcContent = QString::number( QString(CalcTemp.at(0)).toInt() - QString(CalcTemp.at(2)).toInt() );
            }
            else if(CalcTemp.at(1) == "*")
            {
                CalcContent = QString::number( QString(CalcTemp.at(0)).toInt() * QString(CalcTemp.at(2)).toInt() );
            }
            else if(CalcTemp.at(1) == "/")
            {
                CalcContent = QString::number( QString(CalcTemp.at(0)).toInt() / QString(CalcTemp.at(2)).toInt() );
            }
            ui->lineEdit->setText(CalcContent);

        }

    }

    //显示窗口内容
    ui->lineEdit->setText(CalcContent);

}

Widget::~Widget()
{
    delete ui;
}

