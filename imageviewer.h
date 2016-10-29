#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include <QMainWindow>
#include <stack>
#include <QImage> 
#include <QPrinter>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

/* This class is the GUI component of the application.
 * It handles the zooming of images, display and printing.
 * The interfaces to the other classes are the methods : blur() , edgeFilter() , blackWhite()
 */



class SobelFilter;
class Blurrer;
class BWConverter;

class QAction;
class QLabel;
class QMenu;
class QScrollArea;
class QScrollBar;

class ImageViewer : public QMainWindow
{
	Q_OBJECT

	public:
		ImageViewer();
		~ImageViewer();

	private slots: //These are essentially callbacks, invoked when a particular event is triggered on a particular widget.
     
		void open();
		void print();
		void zoomIn();
		void zoomOut();
		void normalSize();
		void fitToWindow();
		void blur();
		void blackWhite();
		void edgeFilter();				
		void undo();
		void redo();
     

	private:
		ImageViewer& operator=(const ImageViewer&);
		ImageViewer(const ImageViewer&);
  
		void createActions();
		void createMenus();
		void updateActions();
		void scaleImage(double factor);
		void adjustScrollBar(QScrollBar *scrollBar, double factor);
		void readImageQLabel(cv::Mat &image);
		void writeImageQLabel(const cv::Mat &input );
		void pushQImageToUndo();
     	 
 
		QLabel *imageLabel;
		QScrollArea *scrollArea;
		double scaleFactor;

 #ifndef QT_NO_PRINTER
		QPrinter printer;
 #endif

		QAction *openAct;
		QAction *printAct;
		QAction *exitAct;
		QAction *zoomInAct;
		QAction *zoomOutAct;
		QAction *normalSizeAct;
		QAction *fitToWindowAct;

		QAction *blurAct;
		QAction *blackWhiteAct;
		QAction *undoAct;
		QAction *redoAct;
		QAction *sobelFilterAct;

		QToolBar *toolbar;
		QMenu *fileMenu;
		QMenu *viewMenu;


		std::stack<QImage *> * undoStack;
		std::stack<QImage *> * redoStack;
		BWConverter * bwConverter;	
		Blurrer * blurrer;	
		SobelFilter * sobelFilter;


     
};

 #endif
