/********************************************************************************
** Form generated from reading UI file 'widget.ui'
**
** Created by: Qt User Interface Compiler version 6.4.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QWidget *widget;
    QVBoxLayout *verticalLayout_2;
    QLineEdit *lineEdit;
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout;
    QPushButton *pushButton_clear;
    QPushButton *pushButton_0;
    QPushButton *pushButton_back;
    QPushButton *pushButton_plus;
    QPushButton *pushButton_1;
    QPushButton *pushButton_2;
    QPushButton *pushButton_3;
    QPushButton *pushButton_minus;
    QPushButton *pushButton_4;
    QPushButton *pushButton_5;
    QPushButton *pushButton_6;
    QPushButton *pushButton_multiply;
    QPushButton *pushButton_7;
    QPushButton *pushButton_8;
    QPushButton *pushButton_9;
    QPushButton *pushButton_divide;
    QPushButton *pushButton_calc;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName("Widget");
        Widget->resize(431, 244);
        widget = new QWidget(Widget);
        widget->setObjectName("widget");
        widget->setGeometry(QRect(10, 10, 399, 211));
        verticalLayout_2 = new QVBoxLayout(widget);
        verticalLayout_2->setObjectName("verticalLayout_2");
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        lineEdit = new QLineEdit(widget);
        lineEdit->setObjectName("lineEdit");
        lineEdit->setMaximumSize(QSize(405, 16777215));

        verticalLayout_2->addWidget(lineEdit);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName("verticalLayout");
        gridLayout = new QGridLayout();
        gridLayout->setObjectName("gridLayout");
        pushButton_clear = new QPushButton(widget);
        pushButton_clear->setObjectName("pushButton_clear");

        gridLayout->addWidget(pushButton_clear, 0, 0, 1, 1);

        pushButton_0 = new QPushButton(widget);
        pushButton_0->setObjectName("pushButton_0");

        gridLayout->addWidget(pushButton_0, 0, 1, 1, 1);

        pushButton_back = new QPushButton(widget);
        pushButton_back->setObjectName("pushButton_back");

        gridLayout->addWidget(pushButton_back, 0, 2, 1, 1);

        pushButton_plus = new QPushButton(widget);
        pushButton_plus->setObjectName("pushButton_plus");

        gridLayout->addWidget(pushButton_plus, 0, 3, 1, 1);

        pushButton_1 = new QPushButton(widget);
        pushButton_1->setObjectName("pushButton_1");

        gridLayout->addWidget(pushButton_1, 1, 0, 1, 1);

        pushButton_2 = new QPushButton(widget);
        pushButton_2->setObjectName("pushButton_2");

        gridLayout->addWidget(pushButton_2, 1, 1, 1, 1);

        pushButton_3 = new QPushButton(widget);
        pushButton_3->setObjectName("pushButton_3");

        gridLayout->addWidget(pushButton_3, 1, 2, 1, 1);

        pushButton_minus = new QPushButton(widget);
        pushButton_minus->setObjectName("pushButton_minus");

        gridLayout->addWidget(pushButton_minus, 1, 3, 1, 1);

        pushButton_4 = new QPushButton(widget);
        pushButton_4->setObjectName("pushButton_4");

        gridLayout->addWidget(pushButton_4, 2, 0, 1, 1);

        pushButton_5 = new QPushButton(widget);
        pushButton_5->setObjectName("pushButton_5");

        gridLayout->addWidget(pushButton_5, 2, 1, 1, 1);

        pushButton_6 = new QPushButton(widget);
        pushButton_6->setObjectName("pushButton_6");

        gridLayout->addWidget(pushButton_6, 2, 2, 1, 1);

        pushButton_multiply = new QPushButton(widget);
        pushButton_multiply->setObjectName("pushButton_multiply");

        gridLayout->addWidget(pushButton_multiply, 2, 3, 1, 1);

        pushButton_7 = new QPushButton(widget);
        pushButton_7->setObjectName("pushButton_7");

        gridLayout->addWidget(pushButton_7, 3, 0, 1, 1);

        pushButton_8 = new QPushButton(widget);
        pushButton_8->setObjectName("pushButton_8");

        gridLayout->addWidget(pushButton_8, 3, 1, 1, 1);

        pushButton_9 = new QPushButton(widget);
        pushButton_9->setObjectName("pushButton_9");

        gridLayout->addWidget(pushButton_9, 3, 2, 1, 1);

        pushButton_divide = new QPushButton(widget);
        pushButton_divide->setObjectName("pushButton_divide");

        gridLayout->addWidget(pushButton_divide, 3, 3, 1, 1);


        verticalLayout->addLayout(gridLayout);

        pushButton_calc = new QPushButton(widget);
        pushButton_calc->setObjectName("pushButton_calc");
        pushButton_calc->setEnabled(true);

        verticalLayout->addWidget(pushButton_calc);


        verticalLayout_2->addLayout(verticalLayout);


        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QCoreApplication::translate("Widget", "Widget", nullptr));
        pushButton_clear->setText(QCoreApplication::translate("Widget", "C", nullptr));
        pushButton_0->setText(QCoreApplication::translate("Widget", "0", nullptr));
        pushButton_back->setText(QCoreApplication::translate("Widget", "->", nullptr));
        pushButton_plus->setText(QCoreApplication::translate("Widget", "+", nullptr));
        pushButton_1->setText(QCoreApplication::translate("Widget", "1", nullptr));
        pushButton_2->setText(QCoreApplication::translate("Widget", "2", nullptr));
        pushButton_3->setText(QCoreApplication::translate("Widget", "3", nullptr));
        pushButton_minus->setText(QCoreApplication::translate("Widget", "-", nullptr));
        pushButton_4->setText(QCoreApplication::translate("Widget", "4", nullptr));
        pushButton_5->setText(QCoreApplication::translate("Widget", "5", nullptr));
        pushButton_6->setText(QCoreApplication::translate("Widget", "6", nullptr));
        pushButton_multiply->setText(QCoreApplication::translate("Widget", "*", nullptr));
        pushButton_7->setText(QCoreApplication::translate("Widget", "7", nullptr));
        pushButton_8->setText(QCoreApplication::translate("Widget", "8", nullptr));
        pushButton_9->setText(QCoreApplication::translate("Widget", "9", nullptr));
        pushButton_divide->setText(QCoreApplication::translate("Widget", "/", nullptr));
        pushButton_calc->setText(QCoreApplication::translate("Widget", "=", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
