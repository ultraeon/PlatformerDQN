import game.*;
import graphics.*;
import java.io.File;
import java.io.FileNotFoundException;
import py4j.GatewayServer;

public class Main {
    public static void main(String[] args) throws InterruptedException, FileNotFoundException {
        EnvironmentInfo env = new EnvironmentInfo();
        GatewayServer server = new GatewayServer(env);
        server.start();
    }
}
