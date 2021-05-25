# Cli Printer

It is very simple to log something inside of the application. Simply write to the cli printer
singleton instance and it will be saved on disk and printed out on the screen

```

[...]

cli = CliPrinter().instance

with cli.header("Header1", 1):
    cli.print("Test")

[...]
```

# Data Logger

The data logger can be used as follows:

```

[...]

logger = DataLogger(mc.data_dir)
mc.add_main('log', logger)

logger.create_field("field1", 1)
logger.log({"field1": [1]})

[...]
```


