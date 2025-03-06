import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public final class TextFileConverter {
    public TextFileConverter() {}
    public static int[][] txtToIntArray(File f) throws FileNotFoundException {
        Scanner s = new Scanner(f);
        s.useDelimiter(",");
        ArrayList<int[]> initResult = new ArrayList<int[]>();
        while(s.hasNextInt()) {
            int[] currObj = new int[5];
            for(int i = 0; i < 5; i++) {
                currObj[i] = s.nextInt();
            }
            System.out.println();
            initResult.add(currObj);
            s.nextLine();
        }
        int[][] result = new int[initResult.size()][5];
        for(int i = 0; i < result.length; i++) {
            result[i] = initResult.get(i).clone();
        }
        s.close();
        return result;
    }
}