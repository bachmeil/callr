import std.stdio;
import embedr.r;

void main() {
  startR();
  alias rvec = RData!RVector;
  alias rlist = RData!RList;
  auto x = rvec("rnorm(12)");
  writeln(x[0]);
  writeln(x[11]);
  printR(x.name ~ "[1]");
  printR(x.name ~ "[12]");
  evalRQ([`set.seed(200)`, `y <- rnorm(100)`, `x <- rnorm(100)`]);
  auto fit = rlist(`lm(y ~ x)`);
  printR(fit.data);
  writeln(fit.names);
  writeln(RVector(fit["coefficients"])[0..2]);
  closeR();
}
