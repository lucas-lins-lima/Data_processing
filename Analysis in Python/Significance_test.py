# pip install pandas numpy scipy openpyxl xlsxwriter

import pandas as pd
import numpy as np
from scipy.stats import norm
import string

relatorio_filtros = []

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

    linhas_para_testar = [i for i in tabela.index if 'Base' not in i]
    for category in linhas_para_testar:

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

def aplicar_filtro_generico(df, data, filtro):
    col = filtro['column']
    prefix = filtro['name']
    label_map = filtro.get('map', None)

    resumo = {
        "prefix": prefix,
        "categorias": {},
        "aplicado": False,
        "erro": False
    }

    # Garante que a coluna de filtro existe no DataFrame original
    if prefix not in df.columns or df[prefix].dropna().empty:
        df[prefix] = df[col].map(label_map) if label_map else df[col]

    # Garante que a coluna de filtro existe também no DataFrame da pergunta
    if col in data.columns:
        data[prefix] = data[col].map(label_map) if label_map else data[col]

    print(f"\n📊 Resumo do filtro: {prefix}")

    # Mostra distribuição do filtro
    if prefix in data.columns and not data[prefix].dropna().empty:
        dist = data[prefix].value_counts(dropna=False)
        for val, count in dist.items():
            val_str = str(val) if pd.notna(val) else "Nulo"
            print(f"- {val_str}: {count}")
            resumo["categorias"][val_str] = count
        print("✔️ Filtro aplicado com sucesso.")
        resumo["aplicado"] = True
    else:
        print("⚠️ Filtro não possui dados válidos para segmentação.")
        print("❌ Filtro com erro.")
        resumo["erro"] = True

    relatorio_filtros.append(resumo)
    return prefix if resumo["aplicado"] else None

def analyze_questions(df, questions, filters):
    resultados = []

    for q in questions:
        question_code = q['code']
        question_text = q['text']
        title = q['title']
        labels = q['labels']
        sheet = q.get("sheet", "Resultados")

        question_filters = [f for f in filters if f.get("sheet").lower() == sheet.lower()]

        print(f"\n🔍 Analisando pergunta: {question_code} - Aba: {sheet}")

        # Adiciona coluna de índice original
        df['_original_index'] = df.index

        if isinstance(question_code, list):
            cols_to_use = question_code + ['_original_index'] + [f['column'] for f in question_filters if f['column'] in df.columns]
            melted = df[cols_to_use].copy()
            melted = melted.melt(
                id_vars=['_original_index'] + [f['column'] for f in question_filters if f['column'] in df.columns],
                value_vars=question_code,
                var_name='opcao', value_name='selecionado'
            )
            melted = melted[melted['opcao'].isin(labels.keys()) & melted['selecionado'].notna()]
            melted['Resposta'] = melted['opcao'].map(labels)
            melted = melted[~melted['Resposta'].isna()]
            data = melted.copy()

            multi_base = df[question_code].notna().any(axis=1)
            total_mean = df.loc[multi_base, question_code].notna().sum(axis=1).mean()
            total_std = df.loc[multi_base, question_code].notna().sum(axis=1).std()
        else:
            data = df[[question_code, '_original_index'] + [f['column'] for f in question_filters if f['column'] in df.columns]].dropna(subset=[question_code]).copy()
            data['Resposta'] = data[question_code].map(labels)
            total_mean = data[question_code].mean()
            total_std = data[question_code].std()

        total_counts = data['Resposta'].value_counts(normalize=True).sort_index() * 100
        total_base = len(data)

        tabela = pd.DataFrame(index=total_counts.index)
        tabela['Total (%)'] = total_counts

        # Detecta se é uma escala de 1 a 5
        disable_box = q.get("disable_box_metrics", False)
        is_likert_1_to_5 = set(labels.keys()) in [{1, 2, 3, 4, 5}, {'1', '2', '3', '4', '5'}]
        if is_likert_1_to_5 and not isinstance(question_code, list) and not disable_box:
            tabela.loc['Topbox (5)', 'Total (%)'] = (data[question_code] == 5).mean() * 100
            tabela.loc['Top2Box (4+5)', 'Total (%)'] = (data[question_code].isin([4, 5])).mean() * 100
            tabela.loc['Bottombox (1)', 'Total (%)'] = (data[question_code] == 1).mean() * 100
            tabela.loc['Bottom2Box (1+2)', 'Total (%)'] = (data[question_code].isin([1, 2])).mean() * 100

        bases = {'Total (%)': total_base}
        medias = {'Total (%)': total_mean}
        desvios = {'Total (%)': total_std}

        for f in question_filters:
            prefix = aplicar_filtro_generico(df, data, f)
            if prefix is None:
                continue

            grouped = data.groupby(prefix)['Resposta'].value_counts(normalize=True).unstack().fillna(0).sort_index(axis=1) * 100
            levels = sorted(data[prefix].dropna().unique())

            for level in levels:
                colname = f'{prefix} {level} (%)'
                if level in grouped.index:
                    tabela[colname] = grouped.loc[level]

                subset_data = data[data[prefix] == level]
                bases[colname] = len(subset_data)

                if isinstance(question_code, list):
                    if not subset_data.empty:
                        ids_filtrados = subset_data['_original_index']
                        linhas_originais = df.loc[ids_filtrados]
                        subset_base = linhas_originais[question_code]
                        multi_valid = subset_base.notna().any(axis=1)
                        if multi_valid.any():
                            mean_val = subset_base.loc[multi_valid].notna().sum(axis=1).mean()
                            std_val = subset_base.loc[multi_valid].notna().sum(axis=1).std()
                        else:
                            mean_val = np.nan
                            std_val = np.nan
                    else:
                        mean_val = np.nan
                        std_val = np.nan
                else:
                    mean_val = subset_data[question_code].mean()
                    std_val = subset_data[question_code].std()

                    if is_likert_1_to_5 and not disable_box:
                        tabela.loc['Topbox (5)', colname] = (subset_data[question_code] == 5).mean() * 100
                        tabela.loc['Top2Box (4+5)', colname] = (subset_data[question_code].isin([4, 5])).mean() * 100
                        tabela.loc['Bottombox (1)', colname] = (subset_data[question_code] == 1).mean() * 100
                        tabela.loc['Bottom2Box (1+2)', colname] = (subset_data[question_code].isin([1, 2])).mean() * 100

                medias[colname] = mean_val
                desvios[colname] = std_val

            # Teste de significância entre grupos (proporções)
            tabela = apply_pairwise_significance(
                tabela,
                data,
                column_prefix=prefix,
                group_column=prefix,
                question_col=question_code
            )

        # Adiciona linhas com métricas finais
        for idx_nome, conteudo in {
            'Base (n)': bases,
            'Média (códigos)': medias,
            'Desvio Padrão': desvios
        }.items():
            for col in tabela.columns:
                val = conteudo.get(col, np.nan)
                tabela.at[idx_nome, col] = val

        # Reorganiza a tabela visualmente
        topo = ['Base (n)', 'Média (códigos)', 'Desvio Padrão']
        box = ['Topbox (5)', 'Top2Box (4+5)', 'Bottombox (1)', 'Bottom2Box (1+2)']

        topo_existente = [i for i in topo if i in tabela.index]
        box_existente = [i for i in box if i in tabela.index]
        respostas = [i for i in tabela.index if i not in topo and i not in box and i != " "]

        if " " not in tabela.index and box_existente:
            tabela.loc[" "] = np.nan

        nova_ordem = topo_existente + respostas + ([" "] if box_existente else []) + box_existente
        tabela = tabela.loc[nova_ordem]

        resultados.append((tabela, title, question_code, question_text, sheet))

    # Remove a coluna auxiliar
    if '_original_index' in df.columns:
        df.drop(columns=['_original_index'], inplace=True)

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
        "code": "Gênero",
        "text": "Gênero. Qual o seu sexo?",
        "title": "Respostas de genero",
        "labels": {
            1: "Masculino",
            2: "Feminino",
            3: "Outro"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Região",
        "text": "Região. Qual a sua localidade?",
        "title": "Respostas de região",
        "labels": {
            1: "Norte",
            2: "Nordeste",
            3: "Centro-Oeste",
            4: "Sudeste",
            5: "Sul"
        },
        "sheet": "Genero_e_usuario",
        "disable_box_metrics": True
    },
    {
        "code": [f"Frases_atributos_{i}" for i in range(1, 13)],
        "text": "Frases_atributos_escolhidas. Quais caracteristicas lhe chamam mais atenção:",
        "title": "Respostas de caracteristicas",
        "labels": {
            "Frases_atributos_1": "Morar bem é uma prioridade pra mim",
            "Frases_atributos_2": "Meu imóvel precisa estar conectado ao meu ritmo de vida moderno",
            "Frases_atributos_3": "Busco um espaço que reflita minha personalidade e estilo",
            "Frases_atributos_4": "Vejo imóveis como uma forma de investimento para o futuro",
            "Frases_atributos_5": "Vejo a Vitacon como uma marca moderna e inovadora",
            "Frases_atributos_6": "Sou alguém que valoriza mobilidade urbana e soluções compactas",
            "Frases_atributos_7": "Valorizo imóveis com soluções que otimizem o meu dinheiro",
            "Frases_atributos_8": "Acho os imóveis da Vitacon com design inteligente e funcional",
            "Frases_atributos_9": "Prefiro pagar por menos espaço se for bem localizado",
            "Frases_atributos_10": "Gosto de marcas que pensam em sustentabilidade e tecnologia",
            "Frases_atributos_11": "Acredito em morar com praticidade, sem abrir mão do conforto",
            "Frases_atributos_12": "Me parece uma marca voltada para quem vive em grandes centros",
            "Frases_atributos_13": "Acredito em soluções compartilhadas e mais conscientes"
        },
        "sheet": "Genero_e_usuario",
        "multi_base": True
    },
    {
        "code": [f"Marcas_conhecidas_{i}" for i in range(1, 11)],
        "text": "Marcas_conhecidas. Quais marcas você conhece?",
        "title": "Respostas de marcas que conhece",
        "labels": {
            "Marcas_conhecidas_1": "Brookfield",
            "Marcas_conhecidas_2": "Cyrela",
            "Marcas_conhecidas_3": "EZTEC",
            "Marcas_conhecidas_4": "Even",
            "Marcas_conhecidas_5": "Lopes",
            "Marcas_conhecidas_6": "MRV",
            "Marcas_conhecidas_7": "Setin",
            "Marcas_conhecidas_8": "Tenda",
            "Marcas_conhecidas_9": "Trisul",
            "Marcas_conhecidas_10": "Vitacon"
        },
        "sheet": "Genero_e_usuario",
        "multi_base": True
    },
    {
        "code": [f"Marcas_frequentes_{i}" for i in range(1, 11)],
        "text": "Marcas_frequentes. Quais marcas você consome mais frequentemente?",
        "title": "Resposta marcas frequentes",
        "labels": {
            "Marcas_frequentes_1": "Brookfield",
            "Marcas_frequentes_2": "Cyrela",
            "Marcas_frequentes_3": "EZTEC",
            "Marcas_frequentes_4": "Even",
            "Marcas_frequentes_5": "Lopes",
            "Marcas_frequentes_6": "MRV",
            "Marcas_frequentes_7": "Setin",
            "Marcas_frequentes_8": "Tenda",
            "Marcas_frequentes_9": "Trisul",
            "Marcas_frequentes_10": "Vitacon"
        },
        "sheet": "Genero_e_usuario",
        "multi_base": True
    },
    {
        "code": [f"Marcas_mais_vendem_{i}" for i in range(1, 11)],
        "text": "Marcas_mais_vendem. Quais marcas você acha que vendem mais?",
        "title": "Resposta de marcas que mais vendem",
        "labels": {
            "Marcas_mais_vendem_1": "Brookfield",
            "Marcas_mais_vendem_2": "Cyrela",
            "Marcas_mais_vendem_3": "EZTEC",
            "Marcas_mais_vendem_4": "Even",
            "Marcas_mais_vendem_5": "Lopes",
            "Marcas_mais_vendem_6": "MRV",
            "Marcas_mais_vendem_7": "Setin",
            "Marcas_mais_vendem_8": "Tenda",
            "Marcas_mais_vendem_9": "Trisul",
            "Marcas_mais_vendem_10": "Vitacon"
        },
        "sheet": "Genero_e_usuario",
        "multi_base": True
    },
    {
        "code": [f"Adjetivos_Vitacon_{i}" for i in range(1, 10)],
        "text": "Adjetivos_Vitacon. Qual a sua opinião sobre os imoveis da Vitacon?",
        "title": "Resposta de adjetivos",
        "labels": {
            "Adjetivos_Vitacon_1": "acessível",
            "Adjetivos_Vitacon_2": "descolado",
            "Adjetivos_Vitacon_3": "elitista",
            "Adjetivos_Vitacon_4": "frio",
            "Adjetivos_Vitacon_5": "funcional",
            "Adjetivos_Vitacon_6": "impessoal",
            "Adjetivos_Vitacon_7": "inovador",
            "Adjetivos_Vitacon_8": "moderno",
            "Adjetivos_Vitacon_9": "ousado"
        },
        "sheet": "Genero_e_usuario",
        "multi_base": True
    },
    {
        "code": [f"Canais_informacao_{i}" for i in range(1, 6)],
        "text": "Canais_informacao. Quais canais de comunicação você procuro sobre imoveis?",
        "title": "Resposta de canais",
        "labels": {
            "Canais_informacao_1": "Corretores",
            "Canais_informacao_2": "Indicação de amigos",
            "Canais_informacao_3": "Portais imobiliários",
            "Canais_informacao_4": "Redes sociais",
            "Canais_informacao_5": "Sites de busca"
        },
        "sheet": "Genero_e_usuario",
        "multi_base": True
    },
    {
        "code": [f"Criterios_escolha_imovel_{i}" for i in range(1, 8)],
        "text": "Criterios_escolha_imovel. Qual criterio você considera mais importante no momento de procurar um imovel?",
        "title": "Respostas de criterios",
        "labels": {
            "Criterios_escolha_imovel_1": "Design",
            "Criterios_escolha_imovel_2": "Localização",
            "Criterios_escolha_imovel_3": "Marca da construtora",
            "Criterios_escolha_imovel_4": "Preço",
            "Criterios_escolha_imovel_5": "Sustentabilidade",
            "Criterios_escolha_imovel_6": "Tamanho",
            "Criterios_escolha_imovel_7": "Áreas comuns"
        },
        "sheet": "Genero_e_usuario",
        "multi_base": True
    },
    {
        "code": "Conhece_Vitacon",
        "text": "Conhece_Vitacon. Você conhece os empreendimentos da Vitacon?",
        "title": "Respostas se conhece a Vitacon",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Usou_Vitacon",
        "text": "Usou_Vitacon. Você já usou algum imovel da Vitacon?",
        "title": "Respostas se já usou Vitcon",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Conhecimento_mercado",
        "text": "Conhecimento_mercado. Você possui conhecimento sobre o mercado imobiliario?",
        "title": "Resposta se possui conhecimento",
        "labels": {
            1: "Baixo",
            2: "Médio",
            3: "Alto"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Interesse_luxo",
        "text": "Interesse_luxo. Você tem interesse em saber sobre o mercado imobiliario de luxo?",
        "title": "Resposta se tem interesse no mercado imobiliario de luxo",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Investimento_imobiliario",
        "text": "Investimento_imobiliario. Você investe no mercado imobiliario?",
        "title": "Resposta se investe",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Finalidade_imovel",
        "text": "Finalidade_imovel. Qual a finaliade do seu interesse nos imoveis da Vitacon?",
        "title": "Resposta no interesse",
        "labels": {
            1: "Investimento",
            2: "Moradia"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Qualidade_percebida",
        "text": "Qualidade_percebida. Para você de 1 a 5 qual a qualidade dos imoveis da Vitacon:",
        "title": "Resposta na qualidade",
        "labels": {
            1: "Muito Ruim",
            2: "Ruim",
            3: "Normal",
            4: "Bom",
            5: "Muito bom"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Custo_beneficio",
        "text": "Custo_beneficio. Para você de 1 a 5 o custo-beneficio dos imoveis na Vitacon é:",
        "title": "Resposta de custo-beneficio",
        "labels": {
            1: "Muito Ruim",
            2: "Ruim",
            3: "Normal",
            4: "Bom",
            5: "Muito bom"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Prob_recomendar",
        "text": "Prob_recomendar. Para você de 1 a 5 qual a probabilidade de voê recomendar os imoveis da Vitacon:",
        "title": "Resposta da probabilidade de recomendar",
        "labels": {
            1: "Com certeza não recomendaria",
            2: "A grande maioria não recomendaria",
            3: "Talvez",
            4: "Alguns recomendaria",
            5: "Com certeza recomendaria"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Gosta_imoveis_Vitacon",
        "text": "Gosta_imoveis_Vitacon. Para você de 1 a 5 o quanto você gosta dos imoveis da Vitacon?",
        "title": "Resposta de gostar dos imoveis",
        "labels": {
            1: "Não gostou nada",
            2: "Não gosto",
            3: "Normal",
            4: "Gostou",
            5: "Gosto muito"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Se_imagina_morando_Vitacon",
        "text": "Se_imagina_morando_Vitacon. Você se imagina morando em um imovel da Vitacon",
        "title": "Resposta se imagina morando",
        "labels": {
            1: "Sim",
            0: "Não",
            2: "Talvez",
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Compraria_Vitacon_futuro",
        "text": "Compraria_Vitacon_futuro. Você compraria algum imovel da Vitacon no futuro?",
        "title": "Resposta se compraria",
        "labels": {
            1: "Sim",
            0: "Não",
            2: "Talvez",
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Ja_buscou_imovel_Vitacon",
        "text": "Ja_buscou_imovel_Vitacon. Você já buscou algum imovel da Vitacon?",
        "title": "Resposta se já buscou",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Faixa_orcamento",
        "text": "Faixa_orcamento. Qual a faixa ideal para você nos imoveis da Vitacon?",
        "title": "Resposta de faixa",
        "labels": {
            1: "Até R$300 mil",
            2: "R$300 a 600 mil",
            3: "R$600 mil a 1 milhão",
            4: "Acima de R$1 milhão"
        },
        "sheet": "Genero_e_usuario"
    },
   {
        "code": "Segue_marcas_redes",
        "text": "Segue_marcas_redes. Você segue as marcas de construtoras, incorporadoras e imobiliarias nas redes sociais?",
        "title": "Respostas se segue nas redes",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Segue_influenciadores_imobiliarios",
        "text": "Segue_influenciadores_imobiliarios. Você segue influenciadores no ramo do mercado imobiliario?",
        "title": "Resposta se segue influencer",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Genero_e_usuario"
    },
   {
        "code": "Considera_nome_construtora",
        "text": "Considera_nome_construtora. Você quando procura um imovel, leva em consideração o nome da construtora?",
        "title": "Resposta se considera a construtora",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Genero_e_usuario"
    },
    {
        "code": "Gênero",
        "text": "Gênero. Qual o seu sexo?",
        "title": "Respostas de genero",
        "labels": {
            1: "Masculino",
            2: "Feminino",
            3: "Outro"
        },
        "sheet": "Idade"
    },
    {
        "code": "Região",
        "text": "Região. Qual a sua localidade?",
        "title": "Respostas de região",
        "labels": {
            1: "Norte",
            2: "Nordeste",
            3: "Centro-Oeste",
            4: "Sudeste",
            5: "Sul"
        },
        "sheet": "Idade"
    },
        {
        "code": [f"Frases_atributos_{i}" for i in range(1, 13)],
        "text": "Frases_atributos_escolhidas. Quais caracteristicas lhe chamam mais atenção:",
        "title": "Respostas de caracteristicas",
        "labels": {
            "Frases_atributos_1": "Morar bem é uma prioridade pra mim",
            "Frases_atributos_2": "Meu imóvel precisa estar conectado ao meu ritmo de vida moderno",
            "Frases_atributos_3": "Busco um espaço que reflita minha personalidade e estilo",
            "Frases_atributos_4": "Vejo imóveis como uma forma de investimento para o futuro",
            "Frases_atributos_5": "Vejo a Vitacon como uma marca moderna e inovadora",
            "Frases_atributos_6": "Sou alguém que valoriza mobilidade urbana e soluções compactas",
            "Frases_atributos_7": "Valorizo imóveis com soluções que otimizem o meu dinheiro",
            "Frases_atributos_8": "Acho os imóveis da Vitacon com design inteligente e funcional",
            "Frases_atributos_9": "Prefiro pagar por menos espaço se for bem localizado",
            "Frases_atributos_10": "Gosto de marcas que pensam em sustentabilidade e tecnologia",
            "Frases_atributos_11": "Acredito em morar com praticidade, sem abrir mão do conforto",
            "Frases_atributos_12": "Me parece uma marca voltada para quem vive em grandes centros",
            "Frases_atributos_13": "Acredito em soluções compartilhadas e mais conscientes"
        },
        "sheet": "Idade",
        "multi_base": True
    },
    {
        "code": [f"Marcas_conhecidas_{i}" for i in range(1, 11)],
        "text": "Marcas_conhecidas. Quais marcas você conhece?",
        "title": "Respostas de marcas que conhece",
        "labels": {
            "Marcas_conhecidas_1": "Brookfield",
            "Marcas_conhecidas_2": "Cyrela",
            "Marcas_conhecidas_3": "EZTEC",
            "Marcas_conhecidas_4": "Even",
            "Marcas_conhecidas_5": "Lopes",
            "Marcas_conhecidas_6": "MRV",
            "Marcas_conhecidas_7": "Setin",
            "Marcas_conhecidas_8": "Tenda",
            "Marcas_conhecidas_9": "Trisul",
            "Marcas_conhecidas_10": "Vitacon"
        },
        "sheet": "Idade",
        "multi_base": True
    },
    {
        "code": [f"Marcas_frequentes_{i}" for i in range(1, 11)],
        "text": "Marcas_frequentes. Quais marcas você consome mais frequentemente?",
        "title": "Resposta marcas frequentes",
        "labels": {
            "Marcas_frequentes_1": "Brookfield",
            "Marcas_frequentes_2": "Cyrela",
            "Marcas_frequentes_3": "EZTEC",
            "Marcas_frequentes_4": "Even",
            "Marcas_frequentes_5": "Lopes",
            "Marcas_frequentes_6": "MRV",
            "Marcas_frequentes_7": "Setin",
            "Marcas_frequentes_8": "Tenda",
            "Marcas_frequentes_9": "Trisul",
            "Marcas_frequentes_10": "Vitacon"
        },
        "sheet": "Idade",
        "multi_base": True
    },
    {
        "code": [f"Marcas_mais_vendem_{i}" for i in range(1, 11)],
        "text": "Marcas_mais_vendem. Quais marcas você acha que vendem mais?",
        "title": "Resposta de marcas que mais vendem",
        "labels": {
            "Marcas_mais_vendem_1": "Brookfield",
            "Marcas_mais_vendem_2": "Cyrela",
            "Marcas_mais_vendem_3": "EZTEC",
            "Marcas_mais_vendem_4": "Even",
            "Marcas_mais_vendem_5": "Lopes",
            "Marcas_mais_vendem_6": "MRV",
            "Marcas_mais_vendem_7": "Setin",
            "Marcas_mais_vendem_8": "Tenda",
            "Marcas_mais_vendem_9": "Trisul",
            "Marcas_mais_vendem_10": "Vitacon"
        },
        "sheet": "Idade",
        "multi_base": True
    },
    {
        "code": [f"Adjetivos_Vitacon_{i}" for i in range(1, 10)],
        "text": "Adjetivos_Vitacon. Qual a sua opinião sobre os imoveis da Vitacon?",
        "title": "Resposta de adjetivos",
        "labels": {
            "Adjetivos_Vitacon_1": "acessível",
            "Adjetivos_Vitacon_2": "descolado",
            "Adjetivos_Vitacon_3": "elitista",
            "Adjetivos_Vitacon_4": "frio",
            "Adjetivos_Vitacon_5": "funcional",
            "Adjetivos_Vitacon_6": "impessoal",
            "Adjetivos_Vitacon_7": "inovador",
            "Adjetivos_Vitacon_8": "moderno",
            "Adjetivos_Vitacon_9": "ousado"
        },
        "sheet": "Idade",
        "multi_base": True
    },
    {
        "code": [f"Canais_informacao_{i}" for i in range(1, 6)],
        "text": "Canais_informacao. Quais canais de comunicação você procuro sobre imoveis?",
        "title": "Resposta de canais",
        "labels": {
            "Canais_informacao_1": "Corretores",
            "Canais_informacao_2": "Indicação de amigos",
            "Canais_informacao_3": "Portais imobiliários",
            "Canais_informacao_4": "Redes sociais",
            "Canais_informacao_5": "Sites de busca"
        },
        "sheet": "Idade",
        "multi_base": True
    },
    {
        "code": [f"Criterios_escolha_imovel_{i}" for i in range(1, 8)],
        "text": "Criterios_escolha_imovel. Qual criterio você considera mais importante no momento de procurar um imovel?",
        "title": "Respostas de criterios",
        "labels": {
            "Criterios_escolha_imovel_1": "Design",
            "Criterios_escolha_imovel_2": "Localização",
            "Criterios_escolha_imovel_3": "Marca da construtora",
            "Criterios_escolha_imovel_4": "Preço",
            "Criterios_escolha_imovel_5": "Sustentabilidade",
            "Criterios_escolha_imovel_6": "Tamanho",
            "Criterios_escolha_imovel_7": "Áreas comuns"
        },
        "sheet": "Idade",
        "multi_base": True
    },
    {
        "code": "Conhece_Vitacon",
        "text": "Conhece_Vitacon. Você conhece os empreendimentos da Vitacon?",
        "title": "Respostas se conhece a Vitacon",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Idade"
    },
    {
        "code": "Usou_Vitacon",
        "text": "Usou_Vitacon. Você já usou algum imovel da Vitacon?",
        "title": "Respostas se já usou Vitcon",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Idade"
    },
    {
        "code": "Conhecimento_mercado",
        "text": "Conhecimento_mercado. Você possui conhecimento sobre o mercado imobiliario?",
        "title": "Resposta se possui conhecimento",
        "labels": {
            1: "Baixo",
            2: "Médio",
            3: "Alto"
        },
        "sheet": "Idade"
    },
    {
        "code": "Interesse_luxo",
        "text": "Interesse_luxo. Você tem interesse em saber sobre o mercado imobiliario de luxo?",
        "title": "Resposta se tem interesse no mercado imobiliario de luxo",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Idade"
    },
    {
        "code": "Investimento_imobiliario",
        "text": "Investimento_imobiliario. Você investe no mercado imobiliario?",
        "title": "Resposta se investe",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Idade"
    },
    {
        "code": "Finalidade_imovel",
        "text": "Finalidade_imovel. Qual a finaliade do seu interesse nos imoveis da Vitacon?",
        "title": "Resposta no interesse",
        "labels": {
            1: "Investimento",
            2: "Moradia"
        },
        "sheet": "Idade"
    },
    {
        "code": "Qualidade_percebida",
        "text": "Qualidade_percebida. Para você de 1 a 5 qual a qualidade dos imoveis da Vitacon:",
        "title": "Resposta na qualidade",
        "labels": {
            1: "Muito Ruim",
            2: "Ruim",
            3: "Normal",
            4: "Bom",
            5: "Muito bom"
        },
        "sheet": "Idade"
    },
    {
        "code": "Custo_beneficio",
        "text": "Custo_beneficio. Para você de 1 a 5 o custo-beneficio dos imoveis na Vitacon é:",
        "title": "Resposta de custo-beneficio",
        "labels": {
            1: "Muito Ruim",
            2: "Ruim",
            3: "Normal",
            4: "Bom",
            5: "Muito bom"
        },
        "sheet": "Idade"
    },
    {
        "code": "Prob_recomendar",
        "text": "Prob_recomendar. Para você de 1 a 5 qual a probabilidade de voê recomendar os imoveis da Vitacon:",
        "title": "Resposta da probabilidade de recomendar",
        "labels": {
            1: "Com certeza não recomendaria",
            2: "A grande maioria não recomendaria",
            3: "Talvez",
            4: "Alguns recomendaria",
            5: "Com certeza recomendaria"
        },
        "sheet": "Idade"
    },
    {
        "code": "Gosta_imoveis_Vitacon",
        "text": "Gosta_imoveis_Vitacon. Para você de 1 a 5 o quanto você gosta dos imoveis da Vitacon?",
        "title": "Resposta de gostar dos imoveis",
        "labels": {
            1: "Não gostou nada",
            2: "Não gosto",
            3: "Normal",
            4: "Gostou",
            5: "Gosto muito"
        },
        "sheet": "Idade"
    },
    {
        "code": "Se_imagina_morando_Vitacon",
        "text": "Se_imagina_morando_Vitacon. Você se imagina morando em um imovel da Vitacon",
        "title": "Resposta se imagina morando",
        "labels": {
            1: "Sim",
            0: "Não",
            2: "Talvez",
        },
        "sheet": "Idade"
    },
    {
        "code": "Compraria_Vitacon_futuro",
        "text": "Compraria_Vitacon_futuro. Você compraria algum imovel da Vitacon no futuro?",
        "title": "Resposta se compraria",
        "labels": {
            1: "Sim",
            0: "Não",
            2: "Talvez",
        },
        "sheet": "Idade"
    },
    {
        "code": "Ja_buscou_imovel_Vitacon",
        "text": "Ja_buscou_imovel_Vitacon. Você já buscou algum imovel da Vitacon?",
        "title": "Resposta se já buscou",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Idade"
    },
    {
        "code": "Faixa_orcamento",
        "text": "Faixa_orcamento. Qual a faixa ideal para você nos imoveis da Vitacon?",
        "title": "Resposta de faixa",
        "labels": {
            1: "Até R$300 mil",
            2: "R$300 a 600 mil",
            3: "R$600 mil a 1 milhão",
            4: "Acima de R$1 milhão"
        },
        "sheet": "Idade"
    },
   {
        "code": "Segue_marcas_redes",
        "text": "Segue_marcas_redes. Você segue as marcas de construtoras, incorporadoras e imobiliarias nas redes sociais?",
        "title": "Respostas se segue nas redes",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Idade"
    },
    {
        "code": "Segue_influenciadores_imobiliarios",
        "text": "Segue_influenciadores_imobiliarios. Você segue influenciadores no ramo do mercado imobiliario?",
        "title": "Resposta se segue influencer",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Idade"
    },
   {
        "code": "Considera_nome_construtora",
        "text": "Considera_nome_construtora. Você quando procura um imovel, leva em consideração o nome da construtora?",
        "title": "Resposta se considera a construtora",
        "labels": {
            1: "Sim",
            0: "Não"
        },
        "sheet": "Idade"
    }
]




filters = [
    {
        "column": "FILTRO_Gênero",
        "name": "FILTRO_Gênero",
        "sheet": "Genero_e_usuario",
        "map": {1: "Masculino", 2: "Feminino", 3:"Outro"}.get
    },
    {
    "column": "FILTRO_Idade",
    "name": "FILTRO_Idade",
    "sheet": "idade",
    "map": {
        1: "18 a 31 anos",
        2: "32 a 44 anos",
        3: "45 a 57 anos",
        4: "58 a 70 anos"
    }.get
},
    {
    "column": "FILTRO_Usou_Vitacon",
    "name": "FILTRO_Usou_Vitacon",
    "sheet": "Genero_e_usuario",
    "map": {
        1: "Usuário Vitacon",
        0: "Usuário concorrente"
    }.get
}
#    {
#        "column": "XX",
#        "name": "xxxxx",
#        "sheet": "xxxxx",
#        "map": lambda x: (
#            "X" if x == 1 else
#            "X" if x in [2, 3] else
#            "X" if x in [4, 5] else
#            "X" if x == 6 else np.nan
#        )
#    },
#    {
#        "column": "XX",
#        "name": "xxxxx",
#        "sheet": "xxxxx",
#        "map": {
#            1: "xxxxx",
#            2: "xxxxx",
#            3: "xxxxx",
#            4: "xxxxx",
#            5: "xxxxx"
#        }.get
#    }
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

    # Formatação
    bold = workbook.add_format({'bold': True})
    number_format = workbook.add_format({'num_format': '0.0', 'align': 'center'})
    center = workbook.add_format({'align': 'center'})
    header_format = workbook.add_format({'bold': True, 'bg_color': '#DCE6F1', 'border': 1, 'align': 'center'})
    note_format = workbook.add_format({'italic': True, 'font_size': 9})
    title_format = workbook.add_format({'bold': True, 'font_size': 12})

    # Dicionários para armazenar resumos por aba
    resumo_por_aba = {}

    for tabela, title, question_code, question_text, sheet in tabelas:
        if sheet not in writer.sheets:
            worksheet = workbook.add_worksheet(sheet)
            writer.sheets[sheet] = worksheet
            aba_offsets[sheet] = 0
            resumo_por_aba[sheet] = {
                'media': [], 'topbox': [], 'top2box': [], 'bottombox': [], 'bottom2box': []
            }
        else:
            worksheet = writer.sheets[sheet]

        row_offset = aba_offsets[sheet]

        # Título e texto da pergunta
        worksheet.merge_range(row_offset, 0, row_offset, len(tabela.columns), f"Tabela {question_code} – {title}", title_format)
        worksheet.merge_range(row_offset + 1, 0, row_offset + 1, len(tabela.columns), question_text)
        worksheet.write(row_offset + 2, 0, 'Fonte: Total da amostra')

        # Cabeçalhos
        col_names = ['Resposta'] + list(tabela.columns)
        for col_num, col_name in enumerate(col_names):
            worksheet.write(row_offset + 4, col_num, col_name, header_format)

        # Conteúdo da tabela
        for r_idx, (index, row) in enumerate(tabela.iterrows(), start=row_offset + 5):
            style = bold if any(p in index for p in ['Base', 'Média', 'Desvio', 'Topbox', 'Top2Box', 'Bottombox', 'Bottom2Box']) else None
            worksheet.write(r_idx, 0, index, style)

            for c_idx, col_name in enumerate(tabela.columns, start=1):
                value = row[col_name]

                if isinstance(value, str):
                    worksheet.write(r_idx, c_idx, value, center)
                elif pd.isna(value) or pd.isnull(value) or (isinstance(value, float) and not np.isfinite(value)):
                    worksheet.write(r_idx, c_idx, "", center)
                elif 'Base' in index:
                    worksheet.write(r_idx, c_idx, value, number_format)
                elif any(m in index for m in ['Média', 'Desvio', 'Topbox', 'Top2Box', 'Bottombox', 'Bottom2Box']):
                    worksheet.write(r_idx, c_idx, value, number_format)
                else:
                    worksheet.write(r_idx, c_idx, f"{value:.1f}%", center)

        # Notas de rodapé
        r_idx += 1
        worksheet.write(r_idx, 0, "A: significant difference at 95%", note_format)
        worksheet.write(r_idx + 1, 0, "a: significant difference at 90%", note_format)
        worksheet.write(r_idx + 2, 0, "Z-Test between two groups", note_format)

        # Armazena resumos se for pergunta com escala 1 a 5 (ou seja, se tiver Topbox)
        if any(m in tabela.index for m in ['Topbox (5)', 'Top2Box (4+5)', 'Bottombox (1)', 'Bottom2Box (1+2)']):
            id_pergunta = f"Tabela {question_code} – {title}"
            linha_media = tabela.loc['Média (códigos)'].copy()
            linha_media.name = id_pergunta
            resumo_por_aba[sheet]['media'].append(linha_media)

            for m, key in [
                ('Topbox (5)', 'topbox'),
                ('Top2Box (4+5)', 'top2box'),
                ('Bottombox (1)', 'bottombox'),
                ('Bottom2Box (1+2)', 'bottom2box')
            ]:
                if m in tabela.index:
                    linha = tabela.loc[m].copy()
                    linha.name = id_pergunta
                    resumo_por_aba[sheet][key].append(linha)

        # Largura das colunas
        worksheet.set_column('A:Z', 20)

        aba_offsets[sheet] = r_idx + 4

    # Ao final de cada aba, insere os resumos se houverem
    for sheet, resumos in resumo_por_aba.items():
        worksheet = writer.sheets[sheet]
        r_idx = aba_offsets[sheet] + 2

        for tipo, label in [
            ('media', 'Resumo – Média (códigos)'),
            ('topbox', 'Resumo – Topbox (5)'),
            ('top2box', 'Resumo – Top2Box (4+5)'),
            ('bottombox', 'Resumo – Bottombox (1)'),
            ('bottom2box', 'Resumo – Bottom2Box (1+2)')
        ]:
            if resumos[tipo]:
                df_resumo = pd.DataFrame(resumos[tipo])
                worksheet.write(r_idx, 0, label, title_format)

                col_names = ['Resposta'] + list(df_resumo.columns)
                for col_num, col_name in enumerate(col_names):
                    worksheet.write(r_idx + 1, col_num, col_name, header_format)

                for i, (idx, row) in enumerate(df_resumo.iterrows()):
                    worksheet.write(r_idx + 2 + i, 0, idx, bold)
                    for j, col in enumerate(df_resumo.columns):
                        val = row[col]
                        if pd.isna(val):
                            worksheet.write(r_idx + 2 + i, j + 1, "", center)
                        else:
                            worksheet.write(r_idx + 2 + i, j + 1, val, number_format)

                r_idx += 4 + len(df_resumo)

        aba_offsets[sheet] = r_idx + 2

    return aba_offsets

# ==========================
# EXECUÇÃO FINAL
# ==========================

arquivo = "Database.xlsx"
df = pd.read_excel(arquivo, sheet_name="Sheet1")

tabelas = analyze_questions(df, questions, filters)

with pd.ExcelWriter("Resultados_T_Test_Vitacon%Com_Teste.xlsx", engine="xlsxwriter") as writer:
    # Exporta as tabelas primeiro
    aba_offsets = export_tables_to_excel(tabelas, writer)
    # Cria a aba de perguntas por último, mas copia ela para o início
    add_question_list_sheet(questions, writer, aba_offsets)
    # Move a aba para a primeira posição
    worksheet_order = list(writer.sheets.keys())
    worksheet_order.insert(0, worksheet_order.pop(worksheet_order.index("Lista de Perguntas")))
    writer.book.worksheets_objs = [writer.sheets[name] for name in worksheet_order]

# ==========================
# RELATÓRIO FINAL DE FILTROS
# ==========================

print("\n📋 RESUMO FINAL DE FILTROS\n" + "="*30)
for r in relatorio_filtros:
    print(f"\n🔎 Filtro: {r['prefix']}")
    for categoria, count in r["categorias"].items():
        print(f" - {categoria}: {count}")
    print(f"✔️ Aplicado: {'Sim' if r['aplicado'] else 'Não'}")
    print(f"❌ Erro: {'Sim' if r['erro'] else 'Não'}")