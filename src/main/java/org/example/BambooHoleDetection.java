package org.example;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class BambooHoleDetection {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static Mat src;
    private static Mat sobelImage;
    private static Mat sobelBinary;
    private static Mat detectedImage;
    private static int thresholdValue = 128;
    private static final int MAX_THRESHOLD_VALUE = 255;

    private static List<File> imageFiles = new ArrayList<>();
    private static int currentIndex = 0;
    private static JLabel originalImageLabel = new JLabel();
    private static JLabel sobelImageLabel = new JLabel();
    private static JLabel sobelBinaryLabel = new JLabel();
    private static JLabel detectedImageLabel = new JLabel();

    public static void main(String[] args) {
        File selectedFolder = openFolderChooser();
        if (selectedFolder == null) {
            System.out.println("未选择文件夹");
            return;
        }

        loadImagesFromFolder(selectedFolder);

        if (imageFiles.isEmpty()) {
            System.out.println("文件夹中没有有效的图像文件");
            return;
        }

        createMainWindow();
    }

    private static void loadImagesFromFolder(File folder) {
        File[] files = folder.listFiles(file -> {
            String name = file.getName().toLowerCase();
            return name.endsWith(".jpg") || name.endsWith(".png") || name.endsWith(".jpeg");
        });

        if (files != null) {
            imageFiles.clear();
            for (File file : files) {
                imageFiles.add(file);
            }
        }
    }

    private static void createMainWindow() {
        JFrame frame = new JFrame("Bamboo Hole Detection");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(1200, 800);

        // 左侧文件列表
        JList<File> fileList = new JList<>(imageFiles.toArray(new File[0]));
        fileList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        fileList.addListSelectionListener(e -> {
            currentIndex = fileList.getSelectedIndex();
            if (currentIndex >= 0) {
                processCurrentImage();
            }
        });

        JScrollPane fileListScrollPane = new JScrollPane(fileList);
        fileListScrollPane.setPreferredSize(new Dimension(200, 0));
        fileListScrollPane.setBorder(BorderFactory.createTitledBorder("图像列表"));

        // 右侧图像显示
        JPanel imagePanel = new JPanel(new GridLayout(2, 2));
        imagePanel.add(wrapInScroll(originalImageLabel, "原图"));
        imagePanel.add(wrapInScroll(sobelImageLabel, "Sobel 图"));
        imagePanel.add(wrapInScroll(sobelBinaryLabel, "Sobel 二值图"));
        imagePanel.add(wrapInScroll(detectedImageLabel, "检测结果"));

        // 底部滑块
        JSlider slider = new JSlider(0, MAX_THRESHOLD_VALUE, thresholdValue);
        slider.setMajorTickSpacing(50);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        slider.addChangeListener(e -> {
            thresholdValue = slider.getValue();
            updateSobelBinary(thresholdValue);
            Image newDetectedImage = updateDetection();
            sobelBinaryLabel.setIcon(new ImageIcon(matToImage(sobelBinary)));
            detectedImageLabel.setIcon(new ImageIcon(newDetectedImage));
        });




        JPanel mainPanel = new JPanel(new BorderLayout());
        mainPanel.add(fileListScrollPane, BorderLayout.WEST);
        mainPanel.add(imagePanel, BorderLayout.CENTER);
        mainPanel.add(slider, BorderLayout.SOUTH);

        frame.add(mainPanel);
        frame.setVisible(true);

        // 显示第一个图像
        if (!imageFiles.isEmpty()) {
            fileList.setSelectedIndex(0);
            processCurrentImage();
        }
    }

    private static void processCurrentImage() {
        File file = imageFiles.get(currentIndex);
        src = Imgcodecs.imread(file.getAbsolutePath());

        if (src.empty()) {
            System.out.println("图像加载失败: " + file.getName());
            return;
        }

        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

        sobelImage = applySobel(gray);
        sobelBinary = new Mat();
        updateSobelBinary(thresholdValue);
        detectedImage = src.clone();
        updateDetection();

        // 更新图像显示
        originalImageLabel.setIcon(new ImageIcon(matToImage(src)));
        sobelImageLabel.setIcon(new ImageIcon(matToImage(sobelImage)));
        sobelBinaryLabel.setIcon(new ImageIcon(matToImage(sobelBinary)));
        detectedImageLabel.setIcon(new ImageIcon(matToImage(detectedImage)));
    }

    private static Mat applySobel(Mat gray) {
        Mat gradX = new Mat();
        Mat gradY = new Mat();
        Mat absGradX = new Mat();
        Mat absGradY = new Mat();
        Mat sobel = new Mat();

        Imgproc.Sobel(gray, gradX, CvType.CV_16S, 1, 0);
        Imgproc.Sobel(gray, gradY, CvType.CV_16S, 0, 1);

        Core.convertScaleAbs(gradX, absGradX);
        Core.convertScaleAbs(gradY, absGradY);

        Core.addWeighted(absGradX, 0.5, absGradY, 0.5, 0, sobel);
        return sobel;
    }

    private static void updateSobelBinary(int threshold) {
        if (sobelBinary == null || sobelBinary.empty()) {
            sobelBinary = new Mat(sobelImage.size(), sobelImage.type());
        }
        Imgproc.threshold(sobelImage, sobelBinary, threshold, MAX_THRESHOLD_VALUE, Imgproc.THRESH_BINARY);
    }

    private static Image updateDetection() {
        detectedImage = src.clone();

        java.util.List<MatOfPoint> contours = new java.util.ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(sobelBinary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > 10 && area < 3000) {
                Rect rect = Imgproc.boundingRect(contour);
                double aspectRatio = (double) rect.width / rect.height;
                if (aspectRatio > 0.5 && aspectRatio < 2.0) {
                    Imgproc.rectangle(detectedImage, rect, new Scalar(0, 255, 0), 2);
                }
            }
        }
        return matToImage(detectedImage);
    }

    private static Image matToImage(Mat mat) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, buffer);
        return Toolkit.getDefaultToolkit().createImage(buffer.toArray());
    }

    private static File openFolderChooser() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("选择包含图像的文件夹");
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        int result = chooser.showOpenDialog(null);
        return result == JFileChooser.APPROVE_OPTION ? chooser.getSelectedFile() : null;
    }

    private static JScrollPane wrapInScroll(JComponent component, String title) {
        JScrollPane scroll = new JScrollPane(component);
        scroll.setBorder(BorderFactory.createTitledBorder(title));
        return scroll;
    }
}
