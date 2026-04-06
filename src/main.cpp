#include "helios/cli.h"

int main(int argc, char** argv) {
    Helios::CLI app(argc, argv);
    if (!app.parse()) {
        return app.exit_code();
    }
    return app.execute();
}
