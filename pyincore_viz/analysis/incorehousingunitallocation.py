# Copyright (c) 2021 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/
import numpy as np
import pandas as pd

class IncoreHousingunitallocation:
    """Utility methods for Housing Unit Allocation Visualization"""
    @staticmethod
    def visualize(dataset, **kwargs):
        """visualize Housing Unit Allocation dataframe.

        Args:
            dataset (obj): Housing unit allocation dataset object.

        Returns:
            None

        """
        hua_df = pd.read_csv(dataset.get_file_path('csv'), header="infer")
        IncoreHousingunitallocation.hua_results_table(hua_df, **kwargs)


    @staticmethod
    def add_race_ethnicity_to_hua_df(df):
        """add race and ethnicity information to HUA dataframe.

        Args:
            df (obj): Pandas DataFrame object.

        Returns:
            object: Pandas DataFrame object.

        """
        df['Race Ethnicity'] = "0 Vacant HU No Race Ethnicity Data"
        df['Race Ethnicity'].notes = "Identify Race and Ethnicity Housing Unit Characteristics."

        df.loc[(df['race'] == 1) & (df['hispan'] == 0), 'Race Ethnicity'] = "1 White alone, Not Hispanic"
        df.loc[(df['race'] == 2) & (df['hispan'] == 0), 'Race Ethnicity'] = "2 Black alone, Not Hispanic"
        df.loc[(df['race'] == 3) & (df['hispan'] == 0), 'Race Ethnicity'] = \
            "3 American Indian and Alaska Native alone, Not Hispanic"
        df.loc[(df['race'] == 4) & (df['hispan'] == 0), 'Race Ethnicity'] = "4 Asian alone, Not Hispanic"
        df.loc[(df['race'].isin([5, 6, 7])) & (df['hispan'] == 0), 'Race Ethnicity'] = "5 Other Race, Not Hispanic"
        df.loc[(df['hispan'] == 1), 'Race Ethnicity'] = "6 Any Race, Hispanic"
        df.loc[(df['gqtype'] >= 1), 'Race Ethnicity'] = "7 Group Quarters no Race Ethnicity Data"
        # Set Race Ethnicity to missing if structure is vacant - makes tables look nicer
        df.loc[(df['Race Ethnicity'] == "0 Vacant HU No Race Ethnicity Data"), 'Race Ethnicity'] = np.nan

        return df

    @staticmethod
    def add_tenure_to_hua_df(df):
        """add tenure information to HUA dataframe.

        Args:
            df (obj): Pandas DataFrame object.

        Returns:
            object: Pandas DataFrame object.

        """
        df['Tenure Status'] = "0 No Tenure Status"
        df['Tenure Status'].notes = "Identify Renter and Owner Occupied Housing Unit Characteristics."

        df.loc[(df['ownershp'] == 1), 'Tenure Status'] = "1 Owner Occupied"
        df.loc[(df['ownershp'] == 2), 'Tenure Status'] = "2 Renter Occupied"
        # Set Tenure Status to missing if structure is vacant - makes tables look nicer
        df.loc[(df['Tenure Status'] == "0 No Tenure Status"), 'Tenure Status'] = np.nan

        return df

    @staticmethod
    def add_colpercent(df, sourcevar, formatedvar):
        """add race and ethnicity information to HUA dataframe.

        Args:
            df (obj): Pandas DataFrame object.
            sourcevar (obj): Pandas Pivottable Column object.
            formatedvar (str): Column name.
        Returns:
            object: Pandas DataFrame object.

        """
        df['%'] = (df[sourcevar] / (df[sourcevar].sum() / 2) * 100)
        df['(%)'] = df.agg('({0[%]:.1f}%)'.format, axis=1)
        df['value'] = df[sourcevar]
        df['format value'] = df.agg('{0[value]:,.0f}'.format, axis=1)
        df[formatedvar] = df['format value'] + '\t ' + df['(%)']

        df = df.drop(columns=[sourcevar, '%', '(%)', 'value', 'format value'])

        return df

    @staticmethod
    def hua_results_table(df, **kwargs):
        """add race and ethnicity information to HUA dataframe.

        Args:
            df (obj): Pandas DataFrame object.
            kwargs (kwargs): Keyword arguments for visualization title.
        Returns:
            object: Pandas DataFrame object.

        """
        who = ""
        what = ""
        when = ""
        where = ""
        row_index = "Race Ethnicity"
        col_index = "Tenure Status"

        if "who" in kwargs.keys():
            who = kwargs["who"]
        if "what" in kwargs.keys():
            what = kwargs["what"]
        if "when" in kwargs.keys():
            when = kwargs["when"]
        if "where" in kwargs.keys():
            where = kwargs["where"]
        if "row_index" in kwargs.keys():
            row_index = kwargs["row_index"]
        if "col_index" in kwargs.keys():
            col_index = kwargs["col_index"]

        df = IncoreHousingunitallocation.add_race_ethnicity_to_hua_df(df)
        df = IncoreHousingunitallocation.add_tenure_to_hua_df(df)

        if who == "Total Households":
            variable = 'numprec'
            function = 'count'
            renamecol = {'Total': who, 'sum': ''}
            num_format = "{:,.0f}"
        elif who == "Total Population":
            variable = 'numprec'
            function = np.sum
            renamecol = {'Total': who, 'sum': ''}
            num_format = "{:,.0f}"
        elif who == "Median Household Income":
            variable = 'randincome'
            function = np.median
            renamecol = {'Total': who}
            num_format = "${:,.0f}"
        else:
            variable = 'numprec'
            function = 'count'
            renamecol = {'Total': who, 'sum': ''}
            num_format = "{:,.0f}"

        # Generate table
        table = pd.pivot_table(df, values=variable, index=[row_index],
                               margins=True, margins_name='Total',
                               columns=[col_index], aggfunc=function).rename(columns=renamecol)
        table_title = "Table. " + who + " " + what + " " + where + " " + when + "."
        varformat = {(who): num_format}
        for col in table.columns:
            varformat[col] = num_format

        # Add Column Percents
        if who in ["Total Households", "Total Population"]:
            for col in table.columns:
                formated_column_name = col + ' (%)'
                table = IncoreHousingunitallocation.add_colpercent(table, col, formated_column_name)

        # Caption Title Style
        styles = [dict(selector="caption",
                       props=[("text-align", "center"),
                              ("caption-side", "top"),
                              ("font-size", "150%"),
                              ("color", 'black')])]    # the color value can not be None

        table = table.style \
            .set_caption(table_title) \
            .set_table_styles(styles) \
            .format(varformat)

        return table
