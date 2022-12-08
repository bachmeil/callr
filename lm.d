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
    y = _y.name;
    x = _x.name;
  }

  void ols() {
    string intTerm = "";
    if (!intercept) {
      intTerm = " -1";
    }
    auto rawFit = r_list(`lm(` ~ y ~ ` ~ ` ~ x ~ intTerm ~ `)`);
    fit = LMFit(rawFit);
    auto rawSummary = r_list(`summary(` ~ rawFit.name ~ `)`);
    summary = LMSummary(rawSummary);
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
