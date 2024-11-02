import java.lang.Exception;
import javax.swing.UIManager;
import java.util.Random;
import java.time.Instant;


public class EcgGen extends javax.swing.JFrame {
    
    /* Main Calculation Objects */
    EcgParam paramOb;
    
    
    /* Main GUI-Window Objects*/
    EcgParamWindow paramWin;
    EcgLogWindow logWin;
    EcgPlotWindow plotWin;
    EcgCalc calcOb;
    EcgExportWindow exportWin;

    /** Creates new form ecgApplication */
    public EcgGen() {
        initClasses();
    }

    /*
     * Init Child Classes
     */
    private void initClasses(){
        // init parameter
        paramOb = new EcgParam();
        logWin = new EcgLogWindow();
        calcOb = new EcgCalc(paramOb, logWin);
        exportWin = new EcgExportWindow(null, true, paramOb, calcOb, logWin);
    }   
    
    public EcgPlotWindow getPlotWin() {
        return plotWin;
    }

    public EcgCalc getCalcOb() {
        return calcOb;
    }

    public EcgParam getParamOb() {
        return paramOb;
    }

    public EcgLogWindow getLog() {
        return logWin;
    }

    public EcgExportWindow getExportWindow() {
        return exportWin;
    }

        /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        
        EcgGen EcgGen = new EcgGen();
        EcgParam paramController = EcgGen.getParamOb();
        EcgLogWindow logger = EcgGen.getLog();
        EcgExportWindow dataExporter = EcgGen.getExportWindow(); 
        EcgCalc generator = EcgGen.getCalcOb();
        
        String activityType = "Resting-Overlap";
        Integer label = 0;
        Integer numSample = 45;
        String destinationLogFolder = "./ECG Generator/Dataset2/Log/";
        String destinationECGFolder = "./ECG Generator/Dataset2/ECG/";


        Random random = new Random();
        Instant timestamp;

        for (int i = 0; i < numSample; i++) {
            System.out.println(Integer.toString(i) + " out of " + Integer.toString(numSample));
            // try {
            //     Thread.sleep(500); // Pauses the program for 1 second (1000 milliseconds)
            // } catch (InterruptedException e) {
            //     e.printStackTrace();
            // }
            timestamp = Instant.now();

            // Warning: use for overlap
            label = random.nextInt(2);

            paramController.resetParameters();
            paramController.setRandomHrStd(activityType, random);
            paramController.setRandomHrMean(activityType, random);
            paramController.setRandomLfHfRatio(activityType, random);
            paramController.setRandomSeed(random);
            paramController.setRandomANoise(random);
            paramController.setRandomFLo(activityType, random);
            paramController.setRandomFHi(activityType, random);
            paramController.setRandomAForR(random);
            paramController.setRandomBForR(random);
            
            if (paramController.checkParameters()) {
                Boolean genSuccess = generator.calculateEcg();
        
                if (genSuccess) {
                    String desFilename = timestamp.toString().replace("/", "") + "_" + Integer.toString(label) + "_" + activityType; // Specify the file path here

                    logger.exportTxtLog(destinationLogFolder + "/" + desFilename + ".txt");
                    dataExporter.exportCsvData(destinationECGFolder + "/" + desFilename + ".csv");
                }
            }
        }
    }
}
