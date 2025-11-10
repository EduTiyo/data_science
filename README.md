1) Missing values
- Campos do tipo contagem (p.ex. `number_of_persons_injured`, `number_of_pedestrians_killed`) são convertidos para inteiro e valores faltantes foram imputados com 0. Motivo: na maior parte dos registros, ausência de valor indica "nenhum ferido/morto reportado". Isso facilita análises agregadas posteriores.
- Campos de data/hora que não puderam ser parseados ficaram com NaN para não introduzir falsos timestamps.

2) Outliers
- Detectamos outliers via IQR para colunas de contagem e, em seguida, aplicamos winsorization (pontos acima do percentil 99 e abaixo do 1 foram capados). Motivo: reduzir influência de valores extremos (possíveis erros de digitação ou eventos raros) sem remover linhas.

3) Inconsistências
- Quando a soma das componentes (pedestrians + cyclists + motorists) excedia o total reportado (`number_of_persons_injured` ou `number_of_persons_killed`), o script atualiza o total para a soma das componentes. Escolha: assumimos que componentes são mais detalhadas e, portanto, mais confiáveis para refletir o total.

4) Padronizações
- `crash_date` -> `crash_date_parsed` (datetime.date) — erros de parse ficam em NaN
- `crash_time` -> `crash_time_parsed` (datetime.time) — flexível para formatos HH:MM, HH:MM:SS e algumas variações
- `zip_code` -> `zip_code_std` (apenas 5 dígitos, zeros à esquerda quando necessário)
- `borough` -> `borough_std` (uppercase)
- `latitude` / `longitude`: coerção para float; valores 0.0 substituídos por NaN (observados no dataset como placeholders inválidos)

5) Logs e artefatos gerados
- `cleaned.csv` — dataset limpo resultante
- `cleaned_cleaning_log.json` — JSON com: percentuais de missing antes e depois, colunas detectadas, decisões aplicadas, número de outliers e correções de inconsistência

6) Observações e possíveis melhorias (próximos passos)
- Reavaliar imputação 0 para contagens com stakeholders; alternativa: marcar como NA e usar imputação estatística quando realmente necessário.
- Investigar valores lat/long nulos e, quando possível, recuperar a localização a partir de campos de endereço.
- Implementar testes unitários (p.ex. pytest) para invariantes das transformações (somas, tipos, formatos).


Se quiser que eu execute agora uma corrida de validação (ex.: `--limit 1000`) e traga o conteúdo do arquivo de log (`cleaned_cleaning_log.json`) com os números reais, posso executar e mostrar o resumo.