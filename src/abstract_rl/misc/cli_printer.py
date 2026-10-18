from os import mkdir
from os.path import exists, join
from types import TracebackType
from typing import Any


class CliPrinter:
    """
    Represents a cli printer, with a specified line length. Can be used to make uniform printing commands.
    """

    class __CliPrinter:
        def __init__(self, line_length: int = 200) -> None:
            """
            Initializes a new cli printer.
            :param line_length: The trajectory length for headers and sections.
            """
            self.line_length = line_length
            self.ver_stack: list[tuple[str, bool]] = [("base", True)]
            self.tabs = 1
            self.filename: str | None = None

        def new_epoch(self, root: str, epoch: int) -> None:
            path = join(root, "logs")
            if not exists(path):
                mkdir(path)
            self.filename = join(path, f"epoch_{epoch}.txt")
            open(self.filename, "w+").close()

        def print_and_write(self, message: str) -> None:
            if self.filename is not None:
                with open(self.filename, "a") as log_file:
                    log_file.write(f"{message}\n")

            print(message)

        def print(self, message: Any) -> Any:
            """
            Prints a new message with a tab in front.
            :param message: The message to print.
            :return: The printer itself.
            """
            if not self.ver_stack[-1][1]:
                return self
            tab_string = "\t" * self.tabs
            self.print_and_write(f"{tab_string}{message}")
            return self

        def single_line(self) -> Any:
            """
            Prints a single line using '-'.
            :return: The printer itself.
            """
            if not self.ver_stack[-1][1]:
                return self
            tab_string = "\t" * self.tabs
            self.print_and_write(tab_string + "-" * (self.line_length - 4 * self.tabs))
            return self

        def dbl_line(self) -> Any:
            """
            Prints a double line using '='
            :return: The printer itself.
            """
            if not self.ver_stack[-1][1]:
                return self
            self.print_and_write("=" * self.line_length)
            return self

        def empty(self) -> Any:
            """
            Prints a double line using '='
            :return: The printer itself.
            """
            if not self.ver_stack[-1][1]:
                return self
            self.print_and_write("")
            return self

        def indent(self, num: int = 1) -> "CliSection":
            return CliSection("indent", self.ver_stack[-1][1], num)

        def big_header(self, title: str, verbose: bool = True) -> "CliSection":
            if verbose:
                self.empty().line(title).dbl_line()
            assert len(self.ver_stack) == 1
            return CliSection(title, verbose, 0)

        def header(self, title: str, verbose: bool = True) -> "CliSection":
            if verbose:
                self.empty().line(title).line()
            return CliSection(title, verbose, 1)

        def line(self, header: str | None = None) -> Any:
            """
            Prints a line with an optional _header.
            :param header the _header to include.
            :return: The printer itself.
            """

            if not self.ver_stack[-1][1]:
                return
            # define the line
            if header is None:
                self.single_line()

            else:
                tl = len(header)
                tab_string = "\t" * self.tabs
                sl = "-" * (self.line_length - tl - 22 - 4 * self.tabs)
                self.print_and_write(f"{tab_string} {header} {sl}")

            return self

    # singleton field
    _instance: Any = None

    def __init__(self, line_length: int = 200) -> None:

        if not CliPrinter._instance:
            CliPrinter._instance = CliPrinter.__CliPrinter(line_length)
        else:
            CliPrinter._instance.line_length = line_length

    @property
    def instance(self) -> Any:
        return self._instance


class CliSection:
    def __init__(self, closure_name: str, verbose: bool = True, indent: int = 0) -> None:
        self.closure_name = closure_name
        self.verbose = verbose
        self.indent = indent

    def __enter__(self) -> None:
        cli = CliPrinter().instance
        cli.ver_stack.append((self.closure_name, self.verbose))
        cli.tabs += self.indent

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        cli = CliPrinter().instance
        assert cli.ver_stack[-1][0] == self.closure_name
        cli.ver_stack = cli.ver_stack[:-1]
        cli.tabs -= self.indent
