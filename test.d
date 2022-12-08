import std.stdio;
import embedr.r;
import callr.lm;

void main() {
  startR();
  alias r_vec = RData!RVector;
  alias r_list = RData!RList;
  auto x = r_vec("rnorm(12)");
  writeln(x[0]);
  writeln(x[11]);
  printR(x.name ~ "[1]");
  printR(x.name ~ "[12]");
  evalRQ([`set.seed(200)`, `y <- rnorm(100)`, `x <- rnorm(100)`]);
  auto fit = r_list(`lm(y ~ x)`);
  printR(fit.data);
  writeln(fit.names);
  writeln(RVector(fit["coefficients"])[0..2]);
  
  // Generate some data
  auto lhs = r_vec("rnorm(100)");
  auto rhs = r_vec("rnorm(100)");
  //~ writeln("lhs name ", lhs.name);
  // Create an LM struct and do OLS
  auto lm = LM(lhs, rhs);
  //~ writeln("lhs name ", lhs.name);
  //~ writeln("rhs name ", rhs.name);
  writeln(lm.y);
  lm.ols();
  lm.fit.coefficients.print("Estimated Coefficients");
  //~ lm.fit.residuals.print("Residuals");
  
  lm.intercept = false;
  lm.ols();
  lm.fit.coefficients.print("New set of coefficients");
  writeln("DF: ", lm.fit.dfResidual);
  closeR();
}
