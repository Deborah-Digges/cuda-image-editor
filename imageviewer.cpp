#include <QtGui>
#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "blur.h"
#include "bw.h"
#include "imageviewer.h"
#include "SobelFilter.h"


/* Image Viewer Constructor.
 * Creates the embedded objects of Blurrer , BWConverter and SobelFilter.
 * Creates the GUI widgets.
 * Creates two stacks of Image Pointers one for the Redo and one for the Undo.
 * Calls createActions and createMenus to initialize the actions and menus of the ImageViewer object.
 */

ImageViewer::ImageViewer()
{
	bwConverter = BWConverter::factory();
	blurrer = Blurrer::factory();
	sobelFilter = SobelFilter::factory();

	imageLabel = new QLabel;
	imageLabel->setBackgroundRole(QPalette::Base);
	imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	imageLabel->setScaledContents(true);

	scrollArea = new QScrollArea;
	scrollArea->setBackgroundRole(QPalette::Dark);
	scrollArea->setWidget(imageLabel); //imageLabel becomes a child of the scroll area.
	setCentralWidget(scrollArea);
	
	undoStack = new std::stack<QImage *>;
	redoStack = new std:: stack<QImage *>;

	createActions();
	createMenus();

	setWindowTitle(tr("Image Viewer"));
	resize(500, 400);
}

/* ImageViewer Destructor
 * It frees the memory for all the members
*/
ImageViewer::~ImageViewer()
{	
	 
	delete imageLabel;
	delete scrollArea;
	delete openAct;
	delete printAct;
	delete exitAct;
	delete zoomInAct;
	delete zoomOutAct;
	delete normalSizeAct;
	delete fitToWindowAct;
	delete blurAct;
	delete blackWhiteAct;
	delete undoAct;
	delete redoAct;
	delete toolbar;
	delete fileMenu;
	delete viewMenu;
	delete undoStack;
	delete redoStack;
	delete bwConverter;	
	delete blurrer;	
}

/* Opens a file dialog box for the user to input a file.
 * Purges the two stacks.
 * Enables the appropriate button widgets after loading the image.
*/
void ImageViewer::open()
{
	while(undoStack->size() != 0)
	{
		undoStack->pop();
	}

        while(redoStack->size() != 0)
        {
                redoStack->pop();
        }

	QString fileName = QFileDialog::getOpenFileName(this , tr("Open File") , QDir::currentPath());
	if (!fileName.isEmpty())
	{
		QImage image(fileName);
		if (image.isNull())
		{
			QMessageBox::information(this, tr("Image Viewer") , tr("Cannot load %1.").arg(fileName));
			return;
		}
		imageLabel->setPixmap(QPixmap::fromImage(image));
		imageLabel->setAlignment (Qt::AlignCenter);
		scaleFactor = 1.0;

		printAct->setEnabled(true);
		fitToWindowAct->setEnabled(true);
		blurAct->setEnabled(true);
		blackWhiteAct->setEnabled(true);
		sobelFilterAct->setEnabled(true);
		redoAct->setEnabled(false);
		undoAct->setEnabled(false);
		
		updateActions();
	

		if (!fitToWindowAct->isChecked())
		{
			imageLabel->adjustSize(); //Enable the label to resize itself : Adjusts the size of the widget to fit the contents.
		}
	}
}

/* Saves the image as a printable form */

void ImageViewer::print()
{
	Q_ASSERT(imageLabel->pixmap());
 	#ifndef QT_NO_PRINTER
	QPrintDialog dialog(&printer, this);
	if (dialog.exec()) 
	{
		QPainter painter(&printer);
		QRect rect = painter.viewport(); 
		QSize size = imageLabel->pixmap()->size();
		size.scale(rect.size(), Qt::KeepAspectRatio);
		painter.setViewport(rect.x(), rect.y(), size.width(), size.height());
		painter.setWindow(imageLabel->pixmap()->rect());
		painter.drawPixmap(0, 0, *imageLabel->pixmap());
	}
 #endif
}

/* Scales the image by a positive value*/

void ImageViewer::zoomIn()
{
	scaleImage(1.25);
}

/* Scales the image by a negative amount */

void ImageViewer::zoomOut()
{
	scaleImage(0.8);
}

/* Scales the image to it's normal size */

void ImageViewer::normalSize()
{
	imageLabel->adjustSize();
	scaleFactor = 1.0;
}

/*Scales the image to fit the window size */

void ImageViewer::fitToWindow()
{
	bool fitToWindow = fitToWindowAct->isChecked();
	scrollArea->setWidgetResizable(fitToWindow);
	if (!fitToWindow)
	{
		normalSize();
	}
	updateActions();
}

/* Creates the action objects for each action of the window
 * Connects the action to the appropriate signals  
 */
void ImageViewer::createActions()
{
	openAct = new QAction(tr("&Open..."), this);
	openAct->setShortcut(tr("Ctrl+O"));
	connect(openAct, SIGNAL(triggered()), this, SLOT(open()));

	printAct = new QAction(tr("&Print..."), this);
	printAct->setShortcut(tr("Ctrl+P"));
	printAct->setEnabled(false);
	connect(printAct, SIGNAL(triggered()), this, SLOT(print()));

	exitAct = new QAction(tr("E&xit"), this);
	exitAct->setShortcut(tr("Ctrl+Q"));
	connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));

	zoomInAct = new QAction(tr("Zoom &In (25%)"), this);
	zoomInAct->setShortcut(tr("Ctrl++"));
	zoomInAct->setEnabled(false);
	connect(zoomInAct, SIGNAL(triggered()), this, SLOT(zoomIn()));

	zoomOutAct = new QAction(tr("Zoom &Out (25%)"), this);
	zoomOutAct->setShortcut(tr("Ctrl+-"));
	zoomOutAct->setEnabled(false);
	connect(zoomOutAct, SIGNAL(triggered()), this, SLOT(zoomOut()));

	normalSizeAct = new QAction(tr("&Normal Size"), this);
	normalSizeAct->setShortcut(tr("Ctrl+S"));
	normalSizeAct->setEnabled(false);
	connect(normalSizeAct, SIGNAL(triggered()), this, SLOT(normalSize()));

	fitToWindowAct = new QAction(tr("&Fit to Window"), this);
	fitToWindowAct->setEnabled(false);
	fitToWindowAct->setCheckable(true);
	fitToWindowAct->setShortcut(tr("Ctrl+F"));
	connect(fitToWindowAct, SIGNAL(triggered()), this, SLOT(fitToWindow()));

}

/* Create menu items 
 * Connect the menu items to appropriate actions.
 */

void ImageViewer::createMenus()
{
	fileMenu = new QMenu(tr("&File"), this);
	fileMenu->addAction(openAct);
	fileMenu->addAction(printAct);
	fileMenu->addSeparator();
	fileMenu->addAction(exitAct);

	viewMenu = new QMenu(tr("&View"), this);
	viewMenu->addAction(zoomInAct);
	viewMenu->addAction(zoomOutAct);
	viewMenu->addAction(normalSizeAct);
	viewMenu->addSeparator();
	viewMenu->addAction(fitToWindowAct);

	menuBar()->addMenu(fileMenu);
	menuBar()->addMenu(viewMenu);
    
	QPixmap bwpix(":/images/bw.jpeg");
	QPixmap blurpix(":/images/blur.jpeg");
	QPixmap undopix(":/images/undo.jpeg");
	QPixmap redopix(":/images/redo.jpeg");
	QPixmap sobelpix(":/images/edge.jpeg");
  
	toolbar = addToolBar("main toolbar");
    
	blurAct = toolbar->addAction(QIcon(blurpix), tr("Blur"));
	blurAct->setEnabled(false);
	connect(blurAct , SIGNAL(triggered()) , this , SLOT(blur()));
    
	blackWhiteAct = toolbar->addAction(QIcon(bwpix) , tr("Black & White"));
	blackWhiteAct->setEnabled(false);
	connect(blackWhiteAct , SIGNAL(triggered()) , this , SLOT(blackWhite()));
	
	sobelFilterAct = toolbar->addAction(QIcon(sobelpix),tr("Edge Detection"));
	sobelFilterAct->setEnabled(false);
	connect(sobelFilterAct , SIGNAL(triggered()), this, SLOT(edgeFilter()));
    
	undoAct = toolbar->addAction(QIcon(undopix),tr("Undo"));
	undoAct->setEnabled(false);
	connect(undoAct , SIGNAL(triggered()), this, SLOT(undo()));

	redoAct = toolbar->addAction(QIcon(redopix),tr("Redo"));
	redoAct->setEnabled(false);
	connect(redoAct , SIGNAL(triggered()), this, SLOT(redo()));

	
}

/* Update actions to set the appropriate widgets enabled */

void ImageViewer::updateActions()
{
	zoomInAct->setEnabled(!fitToWindowAct->isChecked());
	zoomOutAct->setEnabled(!fitToWindowAct->isChecked());
	normalSizeAct->setEnabled(!fitToWindowAct->isChecked());
}


/* Scale Image according to the factor given
 * Image on the QLabel is scaled as desired.
 */
void ImageViewer::scaleImage(double factor)
{
	Q_ASSERT(imageLabel->pixmap());
	scaleFactor *= factor;
	imageLabel->resize(scaleFactor * imageLabel->pixmap()->size());

	adjustScrollBar(scrollArea->horizontalScrollBar(), factor);
	adjustScrollBar(scrollArea->verticalScrollBar(), factor);

	zoomInAct->setEnabled(scaleFactor < 3.0);
	zoomOutAct->setEnabled(scaleFactor > 0.333);
}

/* Helper function to adjust the scroll area */

void ImageViewer::adjustScrollBar(QScrollBar *scrollBar, double factor)
{
	scrollBar->setValue(int(factor * scrollBar->value() + ((factor - 1) * scrollBar->pageStep()/2)));
}

/* Read image from the QLabel in the QImage format.
 * This image is converted into the cv::Mat format needed for processing by the cuda algorithms.
 * The output image to be populated is passed by reference.
 */ 
void ImageViewer::readImageQLabel(cv::Mat &image)
{
	QPixmap pixmap(*(imageLabel->pixmap()));
	QImage current(pixmap.toImage()); 
	current = current.rgbSwapped();
	
	switch (current.format())
	{
		case QImage::Format_RGB888:
		{
			image = cv::Mat(current.height(), current.width(), CV_8UC3 , (uchar*)current.bits(), current.bytesPerLine());
			cv::cvtColor(image, image, CV_RGB2BGR);
			break;
		}
		case QImage::Format_Indexed8:
		{
			image = cv::Mat(current.height(), current.width(), CV_8U , (uchar*)current.bits(), current.bytesPerLine());
			cv::cvtColor(image , image , CV_RGB2BGR);//
			std::cout << "Indexed8\n";
			break;
        
		}
		case QImage::Format_RGB32:
		case QImage::Format_ARGB32:
		case QImage::Format_ARGB32_Premultiplied:
		{
			image = cv::Mat(current.height(), current.width(), CV_8UC4 , (uchar*)current.bits(), current.bytesPerLine());
			
			cv::cvtColor(image,image, CV_RGB2BGR);//
			std::cout << current.format() << "\n";
			break;
        
	        
	    	}
	    	default:
			break;
	}

		
}

/* Takes a cv::Mat image reference as parameter.
 * Converts the image into a QImage format suitable for viewing on the QLabel
 * The image is then displayed on the QLabel.
 */
void ImageViewer::writeImageQLabel(const cv::Mat &input)
{
	
	//cv::Mat imageOutputBGR;
	//cv::cvtColor(input, imageOutputBGR, CV_BGR2RGB);
	QImage result;
	std::cout << "write : " << input.type() << "\n";
	switch(input.type())
	{
        	case CV_8UC3 :
		{
			result = QImage(input.data, input.cols, input.rows, input.step, QImage::Format_RGB888).rgbSwapped().copy();
			std::cout << "8UC3\n";
			break;
		}

	        case CV_8U :
		{
			result = QImage(input.data, input.cols, input.rows, input.step,QImage::Format_Indexed8).rgbSwapped().copy();
			std::cout << "8U\n";
			break;
	        }

		case CV_8UC4 :
		{
			result = QImage(input.data,input.cols, input.rows, input.step , QImage::Format_ARGB32).rgbSwapped().copy();
			std::cout << "8UC4\n";
			break;
           
		}
		default:
		{
			printf("Here!\n");
			result = QImage(input.data, input.cols, input.rows, input.step , QImage::Format_ARGB32).rgbSwapped().copy();
			
			break;
		
		}

        }
	imageLabel->setPixmap(QPixmap::fromImage(result));
	imageLabel->setAlignment (Qt::AlignCenter);
        scaleFactor = 1.0;
	std::cout << "WRITE MAT\n";
}	
 
/* To be called before any change is made to the image on the QLabel
 * Pushes a pointer to a copy of the current image onto the undoStack
 */

void ImageViewer::pushQImageToUndo()
{
	QPixmap pixmap(*(imageLabel->pixmap()));
        undoStack->push(new QImage(pixmap.toImage()));

	if(undoStack->size() == 1)
	{
		undoAct->setEnabled(true);
	}
	std::cout << "Undo : " << undoStack->size() << "\n";
        
}

/* Push a pointer to a copy of the image currently on the QLabel to the redo stack .
 * Obtain the image corresponding to the pointer on the top of the undo stack
 * Write the image to the QLabel
 * Delete the image corresponding to the top of stack pointer.
 * Pop the pointer from the stack
 */  
void ImageViewer::undo()
{
	QImage * current = undoStack->top();
	QPixmap pixmap(*(imageLabel->pixmap()));
	redoStack->push(new QImage(pixmap.toImage()));		
	imageLabel->setPixmap(QPixmap::fromImage(QImage(*current)));
		
	if(redoStack->size() == 1)
	{
		redoAct->setEnabled(true);
	}
	delete undoStack->top();
	undoStack->pop();
	
	if(undoStack->size() == 0)
	{
		undoAct->setEnabled(false);
	}
	if(redoStack->size() == 1)
	{
		redoAct->setEnabled(true);
	}
		
	std::cout << "After undo\n";
	std::cout << "Redo stack : " << redoStack->size() << "\n";
	std::cout << "Undo stack : " << undoStack->size() << "\n";
	
       
}
/* Push a pointer to a copy of the image currently on the QLabel to the undo stack .
 * Obtain the image corresponding to the pointer on top of the redo stack
 * Write the image to the QLabel
 * Free the image corresponding to the pointer
 * Pop the pointer from the redo stack  
 */ 
void ImageViewer::redo()
{
	
	QImage * current = redoStack->top();
	QPixmap pixmap(*(imageLabel->pixmap()));
	undoStack->push(new QImage(pixmap.toImage()));
	imageLabel->setPixmap(QPixmap::fromImage(QImage(*current)));
		
	delete redoStack->top();
	redoStack->pop();
	if(redoStack->size() == 0)
	{
		redoAct->setEnabled(false);
	}
	if(undoStack->size() == 1)
	{
		undoAct->setEnabled(true);
	}
	std::cout << "After redo\n";
	std::cout << "Redo stack : " << redoStack->size() << "\n";
	std::cout << "Undo stack : " << undoStack->size() << "\n";

}

/* Reads the image from the QLabel .
 * Calls the function call operator of the Blurrer embedded object
 */
void ImageViewer::blur()
{	
	Q_ASSERT(imageLabel->pixmap());
	pushQImageToUndo();
	cv::Mat image;
	cv::Mat imageBlurred;
	readImageQLabel(image);
	imageBlurred = (*blurrer)(image);	
	writeImageQLabel(imageBlurred);
		
}

/* Reads the image from the QLabel by calling read.
 * Calls the function call operator of the BWConverter embedded object
 */ 	
void ImageViewer::blackWhite()
{
	Q_ASSERT(imageLabel->pixmap());
	pushQImageToUndo();
	cv::Mat image;		
	cv::Mat imageGrey;
	readImageQLabel(image);
	imageGrey = (*bwConverter)(image);
	writeImageQLabel(imageGrey);

}

/* Reads the image from the QLabel .
 * Calls the function call operator of the SobelClass embedded object
 */
void ImageViewer::edgeFilter()
{
	Q_ASSERT(imageLabel->pixmap());
	pushQImageToUndo();
	
	cv::Mat image;		
	cv::Mat sobelImage;
	readImageQLabel(image);
	sobelImage = (*sobelFilter)(image);
	writeImageQLabel(sobelImage);
}
