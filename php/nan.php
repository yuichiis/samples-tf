<?php
require __DIR__.'/../../rindow-math-matrix/vendor/autoload.php';

$a = log(-1);
echo "log(-1)=".$a."\n";
$b = log(0);
echo "log(0)=".$b."\n";
$c = -log(0);
echo "-log(0)=".$c."\n";

$a = NAN;
assert(is_nan($a));
echo "NAN=".$a."\n";
$a = NAN;
assert(is_nan(-$a));
echo "-NAN=".(-$a)."\n";
$b = INF;
assert(is_infinite($b));
assert($b>0.0);
echo "INF=".$b."\n";
$c = -INF;
assert(is_infinite($c));
assert($c<0.0);
echo "-INF=".$c."\n";

$mo = new Rindow\Math\Matrix\MatrixOperator();
$lacl = $mo->laAccelerated('clblast');
$lacl->blocking(true);
$lacpu = $mo->la();
foreach([$lacl,$lacpu] as $la) {
    $zero = $la->array([-0.0]);
    $minus = $la->array([-1.0]);
    $r = $la->reciprocal($zero);
    echo "div=";var_dump($r->toArray());
    $rlog = $la->log($minus);
    echo "log=";var_dump($rlog->toArray());
    $rsqrt = $la->sqrt($minus);
    echo "sqrt=";var_dump($rsqrt->toArray());
}
