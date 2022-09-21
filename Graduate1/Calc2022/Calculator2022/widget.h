#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

    QString CalcContent;//显示计算内容
    QStringList CalcTemp;//储存运算符等
    void btn_logic(int x, QString i = " ");

private:
    Ui::Widget *ui;
};
#endif // WIDGET_H
