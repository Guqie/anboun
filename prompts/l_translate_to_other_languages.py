l_translate_to_other_languages = """你收到的是一篇中文简报，标题包裹在XML标签<chinese_title>中；内容包裹在XML标签<chinese_content>中。
请将中文简报的标题和内容依次翻译为英语、德语、法语、日语，分别包裹在相应的XML标签中输出。
注意，简报标题和内容必须根据中文版本完整和准确地翻译，同时简报标题必须是一个无标点的连贯句。

输出格式示例：
<english_title>
Here is the brief's English title which must be a coherent sentence containing no punctuation
</english_title>

<english_content>
Here is the brief's English content, which must be a complete and accurate translation of the original Chinese content.
</english_content>

<german_title>
Hier ist der deutsche Titel des Briefs, der ein zusammenhängender Satz ohne Satzzeichen sein muss
</german_title>

<german_content>
Hier ist der deutsche Inhalt der Zusammenfassung, der eine vollständige und genaue Übersetzung des chinesischen Originalinhalts sein muss.
</german_content>

<french_title>
Voici le titre du brief en français, qui doit être une phrase cohérente ne contenant aucune ponctuation
</french_title>

<french_content>
Voici le contenu du rapport en français, qui doit être une traduction complète et exacte du contenu original en chinois.
</french_content>

<japanese_title>
ここに日本語のブリーフのタイトルが入ります。これは句読点を含まない、意味の通る一文でなければなりません
</japanese_title>

<japanese_content>
ここに日本語のブリーフィングの本文が入ります。これは、中国語の原文を完全に、かつ正確に翻訳したものでなければなりません。
</japanese_content>
"""
