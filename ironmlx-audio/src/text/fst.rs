//! OpenFST-compatible execution of the pinned WeText TN graphs.
use crate::{error::invalid, AudioError, Result, SessionControl};
use rustfst::{
    algorithms::{compose::ComposeFst, queues::AutoQueue, tr_filters::AnyTrFilter, Queue},
    prelude::*,
    utils::acceptor,
};
use std::{
    collections::{HashMap, VecDeque},
    path::Path,
};

type Graph = VectorFst<TropicalWeight>;
pub(super) struct WeText {
    zh: [Graph; 2],
    en: [Graph; 2],
}
impl WeText {
    pub fn load(root: &Path) -> Result<Self> {
        fn graphs(root: &Path, lang: &str) -> Result<[Graph; 2]> {
            let mut loaded = Vec::new();
            for name in ["tagger", "verbalizer"] {
                let relative = format!("{lang}/tn/{name}.fst");
                super::verify_resource(
                    &root.join(&relative),
                    "wetext",
                    &format!("wetext/fsts/{relative}"),
                )?;
                loaded.push(
                    Graph::read(root.join(relative)).map_err(|e| invalid("fst", e.to_string()))?,
                );
            }
            Ok(loaded
                .try_into()
                .unwrap_or_else(|_| unreachable!("two graph names")))
        }
        Ok(Self {
            zh: graphs(root, "zh")?,
            en: graphs(root, "en")?,
        })
    }
    pub fn normalize(
        &self,
        text: &str,
        chinese: bool,
        control: &dyn SessionControl,
    ) -> Result<String> {
        let text = text.trim();
        if !regex::Regex::new(r"\d")
            .expect("decimal regex")
            .is_match(text)
        {
            return Ok(text.into());
        }
        let graphs = if chinese { &self.zh } else { &self.en };
        let tagged = transduce(&graphs[0], text, control)?;
        let reordered = reorder(tagged.trim(), chinese)?;
        Ok(transduce(&graphs[1], &reordered, control)?
            .trim()
            .to_owned())
    }
}
fn transduce(fst: &Graph, text: &str, control: &dyn SessionControl) -> Result<String> {
    control.check()?;
    let labels: Vec<u32> = text.bytes().map(u32::from).collect();
    let input: Graph = acceptor(&labels, TropicalWeight::one());
    let composed = ComposeFst::<TropicalWeight, Graph, Graph, _, _, _, _, _>::new_auto(&input, fst)
        .map_err(|e| invalid("fst", e.to_string()))?;
    let start = composed
        .start()
        .ok_or_else(|| invalid("fst", "no matching path"))?;
    let mut graph = Graph::new();
    let first = graph.add_state();
    graph
        .set_start(first)
        .map_err(|e| invalid("fst", e.to_string()))?;
    let mut ids = HashMap::from([(start, first)]);
    let mut queue = VecDeque::from([start]);
    let mut arcs = 0;
    while let Some(state) = queue.pop_front() {
        control.check()?;
        let id = ids[&state];
        if let Some(weight) = composed
            .final_weight(state)
            .map_err(|e| invalid("fst", e.to_string()))?
        {
            graph
                .set_final(id, weight)
                .map_err(|e| invalid("fst", e.to_string()))?;
        }
        let transitions = composed
            .get_trs(state)
            .map_err(|e| invalid("fst", e.to_string()))?;
        arcs += transitions.trs().len();
        if arcs > 1_048_576 {
            return Err(AudioError::CapacityExceeded {
                resource: "text FST arcs",
            });
        }
        for edge in transitions.trs() {
            let next = if let Some(next) = ids.get(&edge.nextstate) {
                *next
            } else {
                if ids.len() >= 262_144 {
                    return Err(AudioError::CapacityExceeded {
                        resource: "text FST states",
                    });
                }
                let next = graph.add_state();
                ids.insert(edge.nextstate, next);
                queue.push_back(edge.nextstate);
                next
            };
            graph
                .add_tr(id, Tr::new(edge.ilabel, edge.olabel, edge.weight, next))
                .map_err(|e| invalid("fst", e.to_string()))?;
        }
    }
    control.check()?;
    rustfst::algorithms::connect(&mut graph).map_err(|e| invalid("fst", e.to_string()))?;
    let bytes = shortest_output(&graph, control)?;
    control.check()?;
    String::from_utf8(bytes).map_err(|e| invalid("fst", e.to_string()))
}
// rustfst 1.3.1 TropicalWeight PartialEq uses KDELTA. OpenFST's single
// shortest-path relaxation compares f32 values exactly. WeText assigns small
// preferences below KDELTA (e.g. "one hundred and one"); preserve those weights.
fn shortest_output(graph: &Graph, control: &dyn SessionControl) -> Result<Vec<u8>> {
    let start = graph
        .start()
        .ok_or_else(|| invalid("fst", "no successful path"))?;
    let mut queue =
        AutoQueue::new(graph, None, &AnyTrFilter {}).map_err(|e| invalid("fst", e.to_string()))?;
    let mut distances = vec![f32::INFINITY; graph.num_states()];
    let mut parents = vec![None; graph.num_states()];
    let mut enqueued = vec![false; graph.num_states()];
    distances[start as usize] = 0.;
    queue.enqueue(start);
    enqueued[start as usize] = true;
    let mut best = f32::INFINITY;
    let mut final_state = None;
    let mut visits = 0;
    while let Some(state) = queue.dequeue() {
        control.check()?;
        visits += 1;
        if visits > 4_194_304 {
            return Err(AudioError::CapacityExceeded {
                resource: "text FST relaxations",
            });
        }
        enqueued[state as usize] = false;
        let distance = distances[state as usize];
        if let Some(weight) = graph
            .final_weight(state)
            .map_err(|e| invalid("fst", e.to_string()))?
        {
            let candidate = distance + weight.value();
            if candidate < best {
                best = candidate;
                final_state = Some(state);
            }
        }
        for (index, edge) in graph
            .get_trs(state)
            .map_err(|e| invalid("fst", e.to_string()))?
            .trs()
            .iter()
            .enumerate()
        {
            let candidate = distance + edge.weight.value();
            let next = edge.nextstate as usize;
            if candidate < distances[next] {
                distances[next] = candidate;
                parents[next] = Some((state, index));
                if !enqueued[next] {
                    queue.enqueue(edge.nextstate);
                    enqueued[next] = true;
                } else {
                    queue.update(edge.nextstate);
                }
            }
        }
    }
    let mut state = final_state.ok_or_else(|| invalid("fst", "no successful path"))?;
    let mut output = Vec::new();
    let mut length = 0;
    while state != start {
        length += 1;
        if length > graph.num_states() {
            return Err(invalid("fst", "cyclic best path"));
        }
        let (parent, index) =
            parents[state as usize].ok_or_else(|| invalid("fst", "incomplete best path"))?;
        let trs = graph
            .get_trs(parent)
            .map_err(|e| invalid("fst", e.to_string()))?;
        let label = trs.trs()[index].olabel;
        if label != 0 {
            output.push(u8::try_from(label).map_err(|_| invalid("fst", "non-byte label"))?);
        }
        state = parent;
    }
    output.reverse();
    Ok(output)
}
// WeText 0.1.2 TokenParser TN ordering. Preserve escaped bytes inside quoted fields.
struct TagParser<'a> {
    text: &'a str,
    position: usize,
}
impl<'a> TagParser<'a> {
    fn whitespace(&mut self) {
        while self.text.as_bytes().get(self.position) == Some(&b' ') {
            self.position += 1;
        }
    }
    fn consume(&mut self, ch: u8) -> Result<()> {
        self.whitespace();
        if self.text.as_bytes().get(self.position) != Some(&ch) {
            return Err(invalid("fst", "malformed tag delimiter"));
        }
        self.position += 1;
        Ok(())
    }
    fn identifier(&mut self) -> Result<&'a str> {
        self.whitespace();
        let start = self.position;
        while self
            .text
            .as_bytes()
            .get(self.position)
            .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
        {
            self.position += 1;
        }
        if start == self.position {
            return Err(invalid("fst", "missing tag name"));
        }
        Ok(&self.text[start..self.position])
    }
    fn quoted(&mut self) -> Result<&'a str> {
        self.consume(b'"')?;
        let start = self.position;
        while let Some(ch) = self.text.as_bytes().get(self.position) {
            if *ch == b'"' {
                let value = &self.text[start..self.position];
                self.position += 1;
                return Ok(value);
            }
            if *ch == b'\\' {
                self.position += 1;
            }
            self.position += 1;
        }
        Err(invalid("fst", "unterminated quoted value"))
    }
}
fn reorder(text: &str, chinese: bool) -> Result<String> {
    let mut parser = TagParser { text, position: 0 };
    let mut output = Vec::new();
    loop {
        parser.whitespace();
        if parser.position == text.len() {
            break;
        }
        let name = parser.identifier()?;
        parser.consume(b'{')?;
        let mut fields = Vec::new();
        loop {
            parser.whitespace();
            if text.as_bytes().get(parser.position) == Some(&b'}') {
                parser.position += 1;
                break;
            }
            let key = parser.identifier()?;
            parser.consume(b':')?;
            let value = parser.quoted()?;
            fields.push((key.to_owned(), value.to_owned()));
        }
        let order: &[&str] = match (chinese, name) {
            (true, "date") => &["year", "month", "day"],
            (true, "fraction") => &["denominator", "numerator"],
            (true, "measure") => &["denominator", "numerator", "value"],
            (true, "money") => &["value", "currency"],
            (true, "time") => &["noon", "hour", "minute", "second"],
            (false, "date") => &["preserve_order", "text", "day", "month", "year"],
            (false, "money") => &[
                "integer_part",
                "fractional_part",
                "quantity",
                "currency_maj",
            ],
            _ => &[],
        };
        let preserve = fields
            .iter()
            .any(|(k, v)| k == "preserve_order" && v == "true");
        let ordered = if order.is_empty() || preserve {
            fields.clone()
        } else {
            order
                .iter()
                .filter_map(|k| fields.iter().find(|(key, _)| key == k).cloned())
                .collect()
        };
        let mut value = format!("{name} {{");
        for (key, field) in ordered {
            value.push_str(&format!(" {key}: \"{field}\""));
        }
        value.push_str(" }");
        output.push(value);
    }
    if output.is_empty() {
        return Err(invalid("fst", "empty or malformed tags"));
    }
    Ok(output.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tags_preserve_quoted_braces_and_escapes() {
        let text = r#"word { name: "a{b}\"c" } money { currency: "元" value: "三" }"#;
        assert_eq!(
            reorder(text, true).unwrap(),
            r#"word { name: "a{b}\"c" } money { value: "三" currency: "元" }"#
        );
        for malformed in [
            "",
            "word {",
            "word { name: nope }",
            r#"word { name: "abc }"#,
        ] {
            assert!(reorder(malformed, true).is_err(), "{malformed}");
        }
    }
    struct Active;
    impl SessionControl for Active {
        fn check(&self) -> Result<()> {
            Ok(())
        }
    }
    #[test]
    fn preserves_sub_delta_weight_preferences() {
        let mut graph = Graph::new();
        let a = graph.add_state();
        let b = graph.add_state();
        let c = graph.add_state();
        graph.set_start(a).unwrap();
        graph.set_final(c, TropicalWeight::one()).unwrap();
        graph
            .add_tr(a, Tr::new(1, b'x' as u32, TropicalWeight::new(1.), c))
            .unwrap();
        graph
            .add_tr(a, Tr::new(1, b'y' as u32, TropicalWeight::new(0.99999), b))
            .unwrap();
        graph
            .add_tr(b, Tr::new(0, 0, TropicalWeight::one(), c))
            .unwrap();
        assert_eq!(shortest_output(&graph, &Active).unwrap(), b"y");
    }
}
