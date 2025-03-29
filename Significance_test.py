# pip install pandas numpy scipy openpyxl xlsxwriter

import pandas as pd
import numpy as np
from scipy.stats import norm
import string

# ==========================
# Funções Utilitárias
# ==========================

def sig_test(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0:
        return ""
    p_comb = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_comb * (1 - p_comb) * (1/n1 + 1/n2))
    if se == 0:
        return ""
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    if p_value < 0.05:
        return "upper"
    elif p_value < 0.10:
        return "lower"
    else:
        return ""

def apply_pairwise_significance(tabela, data, column_prefix, group_column, question_col):
    groups = sorted(data[group_column].dropna().unique())
    group_letters = dict(zip(groups, string.ascii_uppercase))
    prefix = f"{column_prefix} " if column_prefix else ""

    for category in tabela.index[:-2]:
        values = {}
        ns = {}
        sig_labels_upper = {g: [] for g in groups}
        sig_labels_lower = {g: [] for g in groups}

        for group in groups:
            col_name = f"{prefix}{group} (%)"
            try:
                raw_val = tabela.at[category, col_name]
                num_only = ''.join([c for c in str(raw_val) if c.isdigit() or c in ['.', ',']]).replace(',', '.')
                val = float(num_only)
                values[group] = val / 100
                ns[group] = data[data[group_column] == group].shape[0]
            except:
                values[group] = None
                ns[group] = 0

        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if j <= i:
                    continue
                p1, p2 = values[g1], values[g2]
                n1, n2 = ns[g1], ns[g2]
                if p1 is None or p2 is None:
                    continue
                result = sig_test(p1, n1, p2, n2)
                if result == "upper":
                    if p1 > p2:
                        sig_labels_upper[g1].append(group_letters[g2])
                    else:
                        sig_labels_upper[g2].append(group_letters[g1])
                elif result == "lower":
                    if p1 > p2:
                        sig_labels_lower[g1].append(group_letters[g2])
                    else:
                        sig_labels_lower[g2].append(group_letters[g1])

        for group in groups:
            col = f"{prefix}{group} (%)"
            try:
                val = float(''.join([c for c in str(tabela.at[category, col]) if c.isdigit() or c == '.']))
                upper = ''.join(sorted(sig_labels_upper[group]))
                lower = ''.join(sorted(sig_labels_lower[group])).lower()
                letras = upper + lower
                if letras:
                    tabela.at[category, col] = f"{val:.1f}% {letras}"
                else:
                    tabela.at[category, col] = f"{val:.1f}%"
            except:
                continue

    return tabela

def analyze_questions(df, questions, filters):
    resultados = []

    for q in questions:
        question_code = q['code']
        question_text = q['text']
        title = q['title']
        labels = q['labels']
        sheet = q.get("sheet", "Resultados")

        question_filters = [f for f in filters if f.get("sheet") == sheet]

        if isinstance(question_code, list):
            melted = df[question_code + [f['column'] for f in question_filters]].copy()
            melted = melted.melt(id_vars=[f['column'] for f in question_filters], value_vars=question_code,
                                 var_name='opcao', value_name='selecionado')
            melted = melted[melted['opcao'].isin(labels.keys()) & melted['selecionado'].notna()]
            melted['Resposta'] = melted['opcao'].map(labels)
            melted = melted[~melted['Resposta'].isna()]
            data = melted.copy()

            multi_base = df[question_code].notna().any(axis=1)
            total_mean = df.loc[multi_base, question_code].notna().sum(axis=1).mean()
        else:
            data = df[[question_code] + [f['column'] for f in question_filters]].dropna(subset=[question_code]).copy()
            data['Resposta'] = data[question_code].map(labels)
            total_mean = data[question_code].mean()

        total_counts = data['Resposta'].value_counts(normalize=True).sort_index() * 100

        if isinstance(question_code, list):
            total_base = df[question_code].notna().any(axis=1).sum()
        else:
            total_base = data.shape[0]

        tabela = pd.DataFrame(index=total_counts.index)
        tabela['Total (%)'] = total_counts

        bases = {'Total (%)': total_base}
        medias = {'Total (%)': total_mean}

        for f in question_filters:
            col = f['column']
            label_map = f.get('map', None)
            prefix = f['name']

            if label_map:
                data[prefix] = data[col].apply(label_map)
                df[prefix] = df[col].apply(label_map)
            else:
                data[prefix] = data[col]
                df[prefix] = df[col]

            grouped = data.groupby(prefix)['Resposta'].value_counts(normalize=True).unstack().fillna(0).sort_index(axis=1) * 100
            levels = sorted(data[prefix].dropna().unique())

            for level in levels:
                colname = f'{prefix} {level} (%)'
                if level in grouped.index:
                    tabela[colname] = grouped.loc[level]
                subset = df[df[prefix] == level]
                if isinstance(question_code, list):
                    multi_valid = subset[question_code].notna().any(axis=1)
                    mean_val = subset.loc[multi_valid, question_code].notna().sum(axis=1).mean()
                else:
                    mean_val = subset[question_code].mean()
                bases[colname] = subset.shape[0]
                medias[colname] = mean_val

            tabela = apply_pairwise_significance(tabela, data, column_prefix=prefix, group_column=prefix, question_col=question_code)

        tabela.loc['Base (n)'] = pd.Series(bases)
        tabela.loc['Média (códigos)'] = pd.Series(medias)
        tabela = tabela.loc[['Base (n)', 'Média (códigos)'] + [i for i in tabela.index if i not in ['Base (n)', 'Média (códigos)']]]

        resultados.append((tabela, title, question_code, question_text, sheet))

    return resultados

def export_tables_to_excel(tabelas, writer):
    workbook = writer.book
    aba_offsets = {}

    for tabela, title, question_code, question_text, sheet in tabelas:
        if sheet not in writer.sheets:
            worksheet = workbook.add_worksheet(sheet)
            writer.sheets[sheet] = worksheet
            aba_offsets[sheet] = 0
        else:
            worksheet = writer.sheets[sheet]

        row_offset = aba_offsets[sheet]

        bold = workbook.add_format({'bold': True})
        percent_format = workbook.add_format({'num_format': '0.0%', 'align': 'center'})
        number_format = workbook.add_format({'num_format': '0.0', 'align': 'center'})
        center = workbook.add_format({'align': 'center'})
        header_format = workbook.add_format({'bold': True, 'bg_color': '#DCE6F1', 'border': 1, 'align': 'center'})
        note_format = workbook.add_format({'italic': True, 'font_size': 9})
        title_format = workbook.add_format({'bold': True, 'font_size': 12})

        worksheet.merge_range(row_offset, 0, row_offset, len(tabela.columns), f"Tabela {question_code} – {title}", title_format)
        worksheet.merge_range(row_offset + 1, 0, row_offset + 1, len(tabela.columns), question_text)
        worksheet.write(row_offset + 2, 0, 'Fonte: Total da amostra')

        for col_num, col_name in enumerate(tabela.columns.insert(0, 'Resposta')):
            worksheet.write(row_offset + 4, col_num, col_name, header_format)

        for r_idx, (index, row) in enumerate(tabela.iterrows(), start=row_offset + 5):
            worksheet.write(r_idx, 0, index, bold if 'Base' in index or 'Média' in index else None)
            for c_idx, col_name in enumerate(tabela.columns, start=1):
                value = row[col_name]
                if isinstance(value, str):
                    worksheet.write(r_idx, c_idx, value, center)
                elif 'Base' in index:
                    worksheet.write(r_idx, c_idx, value, number_format)
                else:
                    display = f"{value:.1f}%"
                    worksheet.write(r_idx, c_idx, display, center)

        r_idx += 1
        worksheet.write(r_idx, 0, "A: significant difference at 95%", note_format)
        worksheet.write(r_idx + 1, 0, "a: significant difference at 90%", note_format)
        worksheet.write(r_idx + 2, 0, "Z-Test between two groups", note_format)

        worksheet.set_column('A:Z', 20)
        aba_offsets[sheet] = r_idx + 4

# ==========================
# CONFIGURAÇÃO DE PERGUNTAS E FILTROS
# ==========================

questions = [
    {
        "code": "XX",
        "text": "XX. xxxxx",
        "title": "xxxxxx",
        "labels": {
            1: "xxxxx",
            2: "xxxxx",
            3: "xxxxx",
            4: "xxxxx",
            5: "xxxxx",
            6: "xxxxx",
            7: "xxxxx"
        },
        "sheet": "xxxxx"
    },
    {
        "code": "XX",
        "text": "XX. xxxxx",
        "title": "xxxxx",
        "labels": {
            1: "xxxxx",
            2: "xxxxx",
            3: "xxxxx",
            4: "xxxxx"
        },
        "sheet": "xxxxx"
    },
    {
        "code": "XX",
        "text": "XX. xxxxx",
        "title": "xxxxx",
        "labels": {
            1: "xxxxx",
            2: "xxxxx",
            3: "xxxxx",
            4: "xxxxx",
            5: "xxxxx",
            6: "xxxxx",
            7: "xxxxx",
            8: "xxxxx"
        },
        "sheet": "xxxxx"
    },
    {
        "code": [f"XX_op_{i}" for i in range(1, 14)],
        "text": "XXX. xxxxx",
        "title": "xxxxx",
        "labels": {
            "XX_op_1": "xxxxx",
            "XX_op_2": "xxxxx",
            "XX_op_3": "xxxxx",
            "XX_op_4": "xxxxx",
            "XX_op_5": "xxxxx",
            "XX_op_6": "xxxxx",
            "XX_op_7": "xxxxx",
            "XX_op_8": "xxxxx",
            "XX_op_9": "xxxxx",
            "XX_op_10": "xxxxx",
            "XX_op_11": "xxxxx",
            "XX_op_12": "xxxxx",
            "XX_op_13": "xxxxx"
        },
        "sheet": "xxxxx",
        "multi_base": True
    },
    {
        "code": "XX",
        "text": "XX. xxxxx",
        "title": "xxxxx",
        "labels": {
            1: "xxxxx",
            2: "xxxxx",
            3: "xxxxx",
            4: "xxxxx",
            5: "xxxxx",
            6: "xxxxx",
            7: "xxxxx"
        },
        "sheet": "xxxxx"
    }
]

filters = [
    {
        "column": "XX",
        "name": "xxxxx",
        "sheet": "xxxxx",
        "map": {1: "xxxxx", 2: "xxxxx"}.get
    },
    {
        "column": "XX",
        "name": "xxxxx",
        "sheet": "xxxxx",
        "map": lambda x: (
            "X" if x == 1 else
            "X" if x in [2, 3] else
            "X" if x in [4, 5] else
            "X" if x == 6 else np.nan
        )
    },
    {
        "column": "XX",
        "name": "xxxxx",
        "sheet": "xxxxx",
        "map": {
            1: "xxxxx",
            2: "xxxxx",
            3: "xxxxx",
            4: "xxxxx",
            5: "xxxxx"
        }.get
    }
]

# ==========================
# ABA DE LISTA DE PERGUNTAS
# ==========================

def add_question_list_sheet(questions, writer, aba_offsets):
    worksheet = writer.book.add_worksheet("Lista de Perguntas")
    writer.sheets["Lista de Perguntas"] = worksheet

    header_format = writer.book.add_format({"bold": True, "bg_color": "#DCE6F1", "border": 1})
    wrap_format = writer.book.add_format({"text_wrap": True})

    worksheet.write(0, 0, "Código da Pergunta", header_format)
    worksheet.write(0, 1, "Texto da Pergunta", header_format)
    worksheet.write(0, 2, "Título da Tabela", header_format)
    worksheet.write(0, 3, "Abas de Resultado", header_format)

    for idx, q in enumerate(questions, start=1):
        code = ", ".join(q['code']) if isinstance(q['code'], list) else q['code']
        sheet_name = q.get('sheet', 'Resultados')
        row_target = aba_offsets.get(sheet_name, 0) + 1
        link = f"internal:'{sheet_name}'!A{row_target}"

        worksheet.write_url(idx, 0, link, wrap_format, string=code)
        worksheet.write(idx, 1, q['text'], wrap_format)
        worksheet.write(idx, 2, q['title'], wrap_format)
        worksheet.write(idx, 3, sheet_name, wrap_format)

    worksheet.set_column("A:D", 50)

# ==========================
# EXPORTAÇÃO DAS TABELAS
# ==========================

def export_tables_to_excel(tabelas, writer):
    workbook = writer.book
    aba_offsets = {}

    bold = workbook.add_format({'bold': True})
    percent_format = workbook.add_format({'num_format': '0.0%', 'align': 'center'})
    number_format = workbook.add_format({'num_format': '0.0', 'align': 'center'})
    center = workbook.add_format({'align': 'center'})
    header_format = workbook.add_format({'bold': True, 'bg_color': '#DCE6F1', 'border': 1, 'align': 'center'})
    note_format = workbook.add_format({'italic': True, 'font_size': 9})
    title_format = workbook.add_format({'bold': True, 'font_size': 12})

    for tabela, title, question_code, question_text, sheet in tabelas:
        if sheet not in writer.sheets:
            worksheet = workbook.add_worksheet(sheet)
            writer.sheets[sheet] = worksheet
            aba_offsets[sheet] = 0
        else:
            worksheet = writer.sheets[sheet]

        row_offset = aba_offsets[sheet]

        worksheet.merge_range(row_offset, 0, row_offset, len(tabela.columns), f"Tabela {question_code} – {title}", title_format)
        worksheet.merge_range(row_offset + 1, 0, row_offset + 1, len(tabela.columns), question_text)
        worksheet.write(row_offset + 2, 0, 'Fonte: Total da amostra')

        for col_num, col_name in enumerate(tabela.columns.insert(0, 'Resposta')):
            worksheet.write(row_offset + 4, col_num, col_name, header_format)

        for r_idx, (index, row) in enumerate(tabela.iterrows(), start=row_offset + 5):
            worksheet.write(r_idx, 0, index, bold if 'Base' in index or 'Média' in index else None)
            for c_idx, col_name in enumerate(tabela.columns, start=1):
                value = row[col_name]
                if isinstance(value, str):
                    worksheet.write(r_idx, c_idx, value, center)
                elif 'Base' in index:
                    worksheet.write(r_idx, c_idx, value, number_format)
                else:
                    display = f"{value:.1f}%"
                    worksheet.write(r_idx, c_idx, display, center)

        r_idx += 1
        worksheet.write(r_idx, 0, "A: significant difference at 95%", note_format)
        worksheet.write(r_idx + 1, 0, "a: significant difference at 90%", note_format)
        worksheet.write(r_idx + 2, 0, "Z-Test between two groups", note_format)

        worksheet.set_column('A:Z', 20)
        aba_offsets[sheet] = r_idx + 4

    return aba_offsets

# ==========================
# EXECUÇÃO FINAL
# ==========================

arquivo = "Consistencia_Teste Rapido Fragrancia.xlsx"
df = pd.read_excel(arquivo, sheet_name="Consistencia_Teste Rapido Fragr")

tabelas = analyze_questions(df, questions, filters)

with pd.ExcelWriter("relatorio_topline.xlsx", engine="xlsxwriter") as writer:
    # Exporta as tabelas primeiro
    aba_offsets = export_tables_to_excel(tabelas, writer)
    # Cria a aba de perguntas por último, mas copia ela para o início
    add_question_list_sheet(questions, writer, aba_offsets)
    # Move a aba para a primeira posição
    worksheet_order = list(writer.sheets.keys())
    worksheet_order.insert(0, worksheet_order.pop(worksheet_order.index("Lista de Perguntas")))
    writer.book.worksheets_objs = [writer.sheets[name] for name in worksheet_order]