package org.example;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.io.File;

public class BambooHoleDetection {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static Mat src, gray, binary, detectedImage;
    private static int thresholdValue = 128; // 初始阈值
    private static final int maxThresholdValue = 255;

    public static void main(String[] args) {
        File selectedFile = openImageFileChooser();
        if (selectedFile == null) {
            System.out.println("未选择文件");
            return;
        }

        String inputPath = selectedFile.getAbsolutePath();
        src = Imgcodecs.imread(inputPath);

        if (src.empty()) {
            System.out.println("图像读取失败");
            return;
        }

        gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

        binary = new Mat();
        detectedImage = src.clone();

        // 创建主窗口,左侧显示灰度,中间显示二值化,右侧显示结果
        JFrame frame = new JFrame("Threshold Adjustment and Detection");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(1500, 600);

        JLabel grayImageLabel = new JLabel(new ImageIcon(matToImage(gray)));
        JLabel binaryImageLabel = new JLabel(new ImageIcon(updateBinary()));
        JLabel detectedImageLabel = new JLabel(new ImageIcon(updateDetection()));

        JSlider slider = new JSlider(0, maxThresholdValue, thresholdValue);
        slider.setMajorTickSpacing(50);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);

        slider.addChangeListener(e -> {
            thresholdValue = slider.getValue();
            binaryImageLabel.setIcon(new ImageIcon(updateBinary()));
            detectedImageLabel.setIcon(new ImageIcon(updateDetection()));
        });

        JSplitPane horizontalSplit1 = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        JSplitPane horizontalSplit2 = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        horizontalSplit1.setLeftComponent(new JScrollPane(grayImageLabel));
        horizontalSplit1.setRightComponent(new JScrollPane(binaryImageLabel));
        horizontalSplit2.setLeftComponent(horizontalSplit1);
        horizontalSplit2.setRightComponent(new JScrollPane(detectedImageLabel));

        horizontalSplit1.setDividerLocation(500);
        horizontalSplit2.setDividerLocation(1000);

        frame.setLayout(new BorderLayout());
        frame.add(horizontalSplit2, BorderLayout.CENTER);
        frame.add(slider, BorderLayout.SOUTH);

        frame.setVisible(true);
    }

    private static File openImageFileChooser() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("选择竹子图像文件");
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        // 添加文件过滤器，只允许png/jpg/jpeg
        chooser.setAcceptAllFileFilterUsed(false);
        chooser.addChoosableFileFilter(new FileFilter() {
            @Override
            public boolean accept(File f) {
                if (f.isDirectory()) return true;
                String name = f.getName().toLowerCase();
                return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg");
            }
            @Override
            public String getDescription() {
                return "图片文件 (*.png, *.jpg, *.jpeg)";
            }
        });

        // 添加预览面板
        chooser.setAccessory(new ImagePreview(chooser));

        int result = chooser.showOpenDialog(null);
        if (result == JFileChooser.APPROVE_OPTION) {
            return chooser.getSelectedFile();
        }
        return null;
    }

    // 图片预览面板
    static class ImagePreview extends JComponent {
        private static final int PREVIEW_WIDTH = 150;
        private static final int PREVIEW_HEIGHT = 150;
        private ImageIcon thumbnail;
        private File file;

        public ImagePreview(JFileChooser chooser) {
            setPreferredSize(new Dimension(PREVIEW_WIDTH, PREVIEW_HEIGHT));
            chooser.addPropertyChangeListener(evt -> {
                boolean update = false;
                String prop = evt.getPropertyName();

                if (JFileChooser.SELECTED_FILE_CHANGED_PROPERTY.equals(prop)) {
                    file = (File) evt.getNewValue();
                    if (file != null && file.isFile()) {
                        loadImage();
                        update = true;
                    }
                }
                if (update) {
                    repaint();
                }
            });
        }

        private void loadImage() {
            thumbnail = null;
            if (file == null) return;

            // 使用 ImageIcon 自动缩放但不保证等比例，可以自己优化
            ImageIcon tmpIcon = new ImageIcon(file.getPath());
            if (tmpIcon.getIconWidth() > PREVIEW_WIDTH) {
                thumbnail = new ImageIcon(tmpIcon.getImage().getScaledInstance(
                        PREVIEW_WIDTH, -1, Image.SCALE_SMOOTH));
            } else {
                thumbnail = tmpIcon;
            }
        }

        @Override
        protected void paintComponent(Graphics g) {
            if (thumbnail == null) {
                g.setColor(Color.lightGray);
                g.fillRect(0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT);
                g.setColor(Color.black);
                g.drawString("No Preview", PREVIEW_WIDTH / 3, PREVIEW_HEIGHT / 2);
            } else {
                int x = (PREVIEW_WIDTH - thumbnail.getIconWidth()) / 2;
                int y = (PREVIEW_HEIGHT - thumbnail.getIconHeight()) / 2;
                thumbnail.paintIcon(this, g, x, y);
            }
        }
    }

    private static Image updateBinary() {
        Imgproc.threshold(gray, binary, thresholdValue, maxThresholdValue, Imgproc.THRESH_BINARY_INV);
        return matToImage(binary);
    }

    private static Image updateDetection() {
        detectedImage = src.clone();

        java.util.List<MatOfPoint> contours = new java.util.ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

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
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, matOfByte);
        byte[] byteArray = matOfByte.toArray();
        return Toolkit.getDefaultToolkit().createImage(byteArray);
    }
}
