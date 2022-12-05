import std.stdio;
import embedr.r;
import callr.lm;

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
  auto lhs = rvec("rnorm(100)");
  auto rhs = rvec("rnorm(100)");
  auto lm = LM!RVector(lhs, rhs);
  LMFit lmfit = lm.ols();
  lmfit.coefficients.print("Estimated Coefficients");
  lmfit.residuals.print("Residuals");
  lm.intercept = false;
  LMFit lmfit2 = lm.ols();
  lmfit2.coefficients.print("New set of coefficients");
  writeln("DF: ", lmfit2.dfResidual);
  closeR();
}
