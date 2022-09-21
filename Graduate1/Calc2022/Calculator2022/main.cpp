#include "widget.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Widget w;

    w.setWindowTitle("Simple Init Calculator by XHD");

    w.show();
    return a.exec();
}
