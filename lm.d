module callr.lm;

import std.stdio;
import embedr.r;

alias r_vec = RData!RVector;
alias r_mat = RData!RMatrix;
alias r_list = RData!RList;
alias r_array = RData!RArrayInsideR;

struct LM {
  string y;
  string x;
  bool intercept = true;
  LMFit fit;
  LMSummary summary;
  
  this(T1, T2)(T1 _y, T2 _x) {
    writeln(_y.name);
    writeln(_x.name);
    y = _y.name;
    writeln(y);
    writeln("here 1");
    x = _x.name;
    writeln("here 2");
    writeln(x);
    writeln("intercept: ", intercept);
    writeln(y);
    writeln(x);
    writeln("this ", this);
    ols();
    writeln("df ", summary.fstat);
  }

  void ols() {
    string intTerm = "";
    if (!intercept) {
      intTerm = " -1";
    }
    writeln("here");
    auto rawFit = r_list(`lm(` ~ y ~ ` ~ ` ~ x ~ intTerm ~ `)`);
    writeln("rawFit names ", rawFit.names);
    fit = LMFit(rawFit);
    //~ writeln("coefs ", fit.coefficients);
    writeln("rank ", fit.rank);
    auto rawSummary = r_list(`summary(` ~ rawFit.name ~ `)`);
    summary = LMSummary(rawSummary);
    writeln("Finishing OLS");
  }
}

struct LMFit {
  RVector coefficients;
  RVector residuals;
  RVector effects;
  RVector fittedValues;
  int rank;
  int dfResidual;
  
  this(r_list fit) {
    coefficients = RVector(fit["coefficients"]);
    residuals = RVector(fit["residuals"]);
    effects = RVector(fit["effects"]);
    fittedValues = RVector(fit["fitted.values"]);
    rank = fit["rank"].scalar!int;
    dfResidual = fit["df.residual"].scalar!int;
  }
}

struct LMSummary {
  double sigma;
  double rsq;
  double adjrsq;
  RVector fstat;
  RMatrix covUnscaled;
  
  this(r_list fit) {
    sigma = fit["sigma"].scalar;
    rsq = fit["r.squared"].scalar;
    adjrsq = fit["adj.r.squared"].scalar;
    fstat = RVector(fit["fstatistic"]);
    covUnscaled = RMatrix(fit["cov.unscaled"]);
  }
}
