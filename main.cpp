 #include <QApplication>
 #include <QDesktopWidget>
 #include "imageviewer.h"

 int main(int argc, char *argv[])
 {
  int WIDTH = 1000;
  int HEIGHT = 1000;
  int screenWidth;
  int screenHeight;
  	
  QApplication app(argc, argv);
  ImageViewer imageViewer;

  imageViewer.setWindowIcon(QIcon(":/images/index.jpeg"));
	
  QDesktopWidget *desktop = QApplication::desktop();
  screenWidth = desktop->width();
  screenHeight = desktop->height(); 
  imageViewer.resize(screenWidth , screenHeight);
  imageViewer.move( 0, 0 );
  
  
 #if defined(Q_OS_SYMBIAN)
     
     imageViewer.showMaximized();
 #else
     imageViewer.show();
 #endif     
  return app.exec();
 }
