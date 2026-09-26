# hgp-lib performance report

## evaluation_default_100_literals_1_000_samples

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="2">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">15.27 ms</td>
      <td align="right">-</td>
      <td align="right">17.39 ms</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">12.50 ms</td>
      <td align="right">-18.1% faster</td>
      <td align="right">12.36 ms</td>
      <td align="right">-28.9% faster</td>
    </tr>
  </tbody>
</table>

## evaluation_default_100_literals_10_000_samples

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="2">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">54.68 ms</td>
      <td align="right">-</td>
      <td align="right">85.24 ms</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">18.74 ms</td>
      <td align="right">-65.7% faster</td>
      <td align="right">21.28 ms</td>
      <td align="right">-75.0% faster</td>
    </tr>
  </tbody>
</table>

## evaluation_default_1_000_literals_1_000_samples

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="2">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">171.34 ms</td>
      <td align="right">-</td>
      <td align="right">184.09 ms</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">132.61 ms</td>
      <td align="right">-22.6% faster</td>
      <td align="right">125.00 ms</td>
      <td align="right">-32.1% faster</td>
    </tr>
  </tbody>
</table>

## evaluation_default_1_000_literals_10_000_samples

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="2">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">2.01 s</td>
      <td align="right">-</td>
      <td align="right">783.25 ms</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">203.24 ms</td>
      <td align="right">-89.9% faster</td>
      <td align="right">223.84 ms</td>
      <td align="right">-71.4% faster</td>
    </tr>
  </tbody>
</table>

## full_run.banknote_authentication.500_epochs

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="3">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">13.94 s</td>
      <td align="right">-</td>
      <td align="right">0.9650 +/- 0.0069</td>
      <td align="right">10.74 s</td>
      <td align="right">-</td>
      <td align="right">0.9662 +/- 0.0093</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">10.39 s</td>
      <td align="right">-25.5% faster</td>
      <td align="right">0.9650 +/- 0.0069</td>
      <td align="right">10.55 s</td>
      <td align="right">-1.8% faster</td>
      <td align="right">0.9662 +/- 0.0093</td>
    </tr>
  </tbody>
</table>

## full_run.breast_cancer.500_epochs

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="3">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">12.15 s</td>
      <td align="right">-</td>
      <td align="right">0.9217 +/- 0.0211</td>
      <td align="right">9.40 s</td>
      <td align="right">-</td>
      <td align="right">0.9383 +/- 0.0154</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">9.82 s</td>
      <td align="right">-19.2% faster</td>
      <td align="right">0.9217 +/- 0.0211</td>
      <td align="right">9.04 s</td>
      <td align="right">-3.8% faster</td>
      <td align="right">0.9383 +/- 0.0154</td>
    </tr>
  </tbody>
</table>

## full_run.diabetes.500_epochs

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="3">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">14.96 s</td>
      <td align="right">-</td>
      <td align="right">0.6203 +/- 0.0543</td>
      <td align="right">11.14 s</td>
      <td align="right">-</td>
      <td align="right">0.6213 +/- 0.0562</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">11.65 s</td>
      <td align="right">-22.1% faster</td>
      <td align="right">0.6203 +/- 0.0543</td>
      <td align="right">10.50 s</td>
      <td align="right">-5.7% faster</td>
      <td align="right">0.6213 +/- 0.0562</td>
    </tr>
  </tbody>
</table>

## full_run.ionosphere.500_epochs

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="3">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">12.08 s</td>
      <td align="right">-</td>
      <td align="right">0.9421 +/- 0.0172</td>
      <td align="right">8.58 s</td>
      <td align="right">-</td>
      <td align="right">0.9200 +/- 0.0178</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">8.80 s</td>
      <td align="right">-27.2% faster</td>
      <td align="right">0.9421 +/- 0.0172</td>
      <td align="right">8.21 s</td>
      <td align="right">-4.3% faster</td>
      <td align="right">0.9200 +/- 0.0178</td>
    </tr>
  </tbody>
</table>

## full_run.spambase.500_epochs

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="3">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">21.17 s</td>
      <td align="right">-</td>
      <td align="right">0.8743 +/- 0.0161</td>
      <td align="right">21.45 s</td>
      <td align="right">-</td>
      <td align="right">0.8787 +/- 0.0082</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">13.71 s</td>
      <td align="right">-35.2% faster</td>
      <td align="right">0.8743 +/- 0.0161</td>
      <td align="right">13.13 s</td>
      <td align="right">-38.8% faster</td>
      <td align="right">0.8787 +/- 0.0082</td>
    </tr>
  </tbody>
</table>

## rule_evaluation.banknote_authentication.100_rules

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">1.73 s</td>
      <td align="right">-</td>
      <td align="right">0.8564 +/- 0.0472</td>
      <td align="right">1.87 s</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">1.68 s</td>
      <td align="right">-2.9% faster</td>
      <td align="right">0.8564 +/- 0.0472</td>
      <td align="right">1.87 s</td>
      <td align="right">-0.4% faster</td>
    </tr>
  </tbody>
</table>

## rule_evaluation.breast_cancer.100_rules

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">892.69 ms</td>
      <td align="right">-</td>
      <td align="right">0.8827 +/- 0.0268</td>
      <td align="right">1.18 s</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">867.29 ms</td>
      <td align="right">-2.8% faster</td>
      <td align="right">0.8827 +/- 0.0268</td>
      <td align="right">1.18 s</td>
      <td align="right">+0.1% slower</td>
    </tr>
  </tbody>
</table>

## rule_evaluation.diabetes.100_rules

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">417.23 ms</td>
      <td align="right">-</td>
      <td align="right">0.5987 +/- 0.0385</td>
      <td align="right">414.16 ms</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">408.11 ms</td>
      <td align="right">-2.2% faster</td>
      <td align="right">0.5987 +/- 0.0385</td>
      <td align="right">414.67 ms</td>
      <td align="right">+0.1% slower</td>
    </tr>
  </tbody>
</table>

## rule_evaluation.ionosphere.100_rules

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">277.49 ms</td>
      <td align="right">-</td>
      <td align="right">0.8799 +/- 0.0338</td>
      <td align="right">349.59 ms</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">270.04 ms</td>
      <td align="right">-2.7% faster</td>
      <td align="right">0.8799 +/- 0.0338</td>
      <td align="right">344.44 ms</td>
      <td align="right">-1.5% faster</td>
    </tr>
  </tbody>
</table>

## rule_evaluation.spambase.100_rules

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">6.74 s</td>
      <td align="right">-</td>
      <td align="right">0.7822 +/- 0.0186</td>
      <td align="right">3.18 s</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">6.59 s</td>
      <td align="right">-2.3% faster</td>
      <td align="right">0.7822 +/- 0.0186</td>
      <td align="right">3.16 s</td>
      <td align="right">-0.4% faster</td>
    </tr>
  </tbody>
</table>

## scoring_default_1_000

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">0.36 ms</td>
      <td align="right">-</td>
      <td align="right">0.5047 +/- 0.0204</td>
      <td align="right">0.32 ms</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">0.36 ms</td>
      <td align="right">+0.4% slower</td>
      <td align="right">0.5047 +/- 0.0204</td>
      <td align="right">0.31 ms</td>
      <td align="right">-0.7% faster</td>
    </tr>
  </tbody>
</table>

## scoring_default_10_000

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">1.51 ms</td>
      <td align="right">-</td>
      <td align="right">0.5007 +/- 0.0062</td>
      <td align="right">1.50 ms</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">1.48 ms</td>
      <td align="right">-2.3% faster</td>
      <td align="right">0.5007 +/- 0.0062</td>
      <td align="right">1.49 ms</td>
      <td align="right">-0.5% faster</td>
    </tr>
  </tbody>
</table>

## scoring_default_100_000

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">14.00 ms</td>
      <td align="right">-</td>
      <td align="right">0.5005 +/- 0.0019</td>
      <td align="right">11.78 ms</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">14.04 ms</td>
      <td align="right">+0.3% slower</td>
      <td align="right">0.5005 +/- 0.0019</td>
      <td align="right">11.83 ms</td>
      <td align="right">+0.4% slower</td>
    </tr>
  </tbody>
</table>

## scoring_default_1_000_000

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="3">hp_1_win</th>
      <th colspan="2">macbook-m3</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>vs previous</th>
      <th>Result</th>
      <th>Time</th>
      <th>vs previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.2.2</td>
      <td align="right">415.48 ms</td>
      <td align="right">-</td>
      <td align="right">0.4996 +/- 0.0006</td>
      <td align="right">97.81 ms</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td>1.2.3</td>
      <td align="right">439.58 ms</td>
      <td align="right">+5.8% slower</td>
      <td align="right">0.4996 +/- 0.0006</td>
      <td align="right">96.96 ms</td>
      <td align="right">-0.9% faster</td>
    </tr>
  </tbody>
</table>
